import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
logger = logging.getLogger(__name__)
from time import perf_counter
from scipy.stats import norm, uniform, multivariate_normal
from scipy.linalg import toeplitz
from pandas import DataFrame
import pandas as pd
import os 

def pretty_print_seconds(secs):
    """I didn't waste that much time on this"""

    secs = int(secs) # perf_counter gives fractional seconds
    if secs == 0:
        return "0secs"
    
    mins, secs = divmod(secs, 60)
    hours, mins = divmod(mins, 60)
    return " ".join( f"{val}{name}" for name, val 
                   in {"hours":hours, "mins":mins, "secs":secs}.items()
                   if val !=0
                   )

def read_df_file(f_name, folder = "logs_and_data"):
    """convenience func that reads `f_name` in logs_and_data folder as a pandas Dataframe"""
    f = os.path.join(folder, f_name)
    d = pd.read_csv(f, index_col = 0)
    return d

def estimate_TV_upper(lag, taus, ts):
    """
    estimate the function `max(0, (tau-lag-t)/lag )` for given `ts` using the sampled `taus` at a specific `lag`
    """
    ests = np.zeros((len(taus), len(ts)))
    zeros = np.zeros(ts.shape)
    # for each individual sample compute the function
    for i, tau in enumerate(taus):
        
        #gives back an array - the func applied to many t for one tau
        ests[i] = np.maximum.reduce([#an element-wise max of two arrays - one all zeros
            zeros, np.ceil((tau-lag-ts)/lag )
                ] )
                    
    return ests.mean(0) #an average of the individual realisations of the upper bound at each t
        
def make_cov_haar(seed, p, scale = 1):
    """generates a random `P`d diagonal covariance matrix    """
    s_sqrt = norm.rvs( size = p**2, random_state= seed).reshape(p,p)
    cov = np.matmul(s_sqrt, s_sqrt)
    return cov/scale

def make_cov_equivar(p, rho, scale = 1):
    """Equicorrelated covariance"""
    return (np.diagflat([1-rho for _ in range(p)]) + rho ) /scale

def make_cov_ar1(seed, p, rho = .5):
    rng = np.random.default_rng(seed)

    perm = np.identity(p)
    #shuffle columns
    rng.shuffle(perm.T) #dont need transpose here as perm is symmetric

    t = toeplitz([rho**i for i in range(p)])
    return np.matmul( perm.T, np.matmul(t, perm) )

def make_cov_diag( p, scale = 1):
    """diagonal covariance"""
    return np.identity(p)/scale

def quad_form_mvn(mu, sigma_inverted, state)->float:
    """The bit in the exponential in the likelihood of a MVN"""
    centred = state - mu
    return np.matmul(np.matmul( centred.T, sigma_inverted), centred)

def make_timestamp():
    """makes a timestamp of roughly right now ( to the minute) as a string"""
    import datetime
    return '{:%Y-%m-%d %a %H-%M}'.format(datetime.datetime.now())

def save_df_with_timestamp(df:DataFrame, msg = "data"):
    """write a dataframe to a csv file with a timestamp and `msg` into the logs_and_data folder.
    
    returns the filename
    """
    target_dir = os.path.join(os.getcwd(), "logs_and_data")
    f_name = msg + " " + make_timestamp() +".csv"
    f_path = os.path.join(target_dir, f_name)

    if not os.path.exists(target_dir):
        #sometimes the logging folder mightn't be on the remote machine
        os.mkdir(target_dir)

    df.to_csv(f_path, index = True)

    return f_name

def estimate_TV_from_file(f_name,num_ts = 100, save_msg = "TV est"):
    """
    estimates the TV upper bound per lag from tau_lag estimates in a file
    looks for the file `f_name` in logs_and_data folder

    Saves the estimates in a timestamped file with `save_msg` and returns them
    """
    df = read_df_file(f_name)

    ts = np.array(range(1,num_ts+1))
    # estimating func is applied along columns , lag stored in column name
    tv_estimates = df.apply(lambda col : estimate_TV_upper(int(col.name), df[col.name], ts)
             , axis = 0)
    tv_estimates.set_index(ts, inplace=True)

    ests_f = save_df_with_timestamp(tv_estimates, save_msg)

    return ests_f

def print_basic_df_summary(df):
    """helper function that prints the mean, std of a dataframe"""
    print( df.apply(["mean", "std"]).agg(round, 0, 2) )

def coupled_MCMC2(lag:int,  max_t_iterations=10**3):
    """
    coupling with a lag
    target a normal with a hardcoded mean and sd"""
        #start timing here
    start_time = perf_counter()

    #initialisation
    x_chain = np.zeros(max_t_iterations)
    y_chain = np.zeros(max_t_iterations)
    x_chain[0], y_chain[0] = -100, 100
    log_unifs = np.log(uniform.rvs(size = max_t_iterations+1)) #theres one spare here just to keep indexing simple

    
        
    def proposal_dist_logpdf(current_state):
        return norm(current_state, 1).logpdf
    def proposal_dist_sampler(current_state):
        return norm(current_state, 1).rvs


    def log_alpha(current, new):
        """log of the alpha probability of accepting a proposed move.
        Here the proposal dist is symmetric and the target is N(3,4)        
        """
        mu = 3
        sigma_squared = 4   
        r = (current - new)*( current+new- 2*mu )/(2*sigma_squared)
        return min(0, r )
    
    # run X chain for lag steps
    for t in range(1,lag+1):
        current_state = x_chain[t-1]
        proposed_state = proposal_dist_sampler(current_state)() # looks ugly i know

        if log_unifs[t] <= log_alpha(current_state, proposed_state):
            x_chain[t] = proposed_state
        else:
            x_chain[t] = current_state
    
    meeting_time = None
    # now run a coupling with the lagged chains
    for t in range(lag+1, max_t_iterations):
        current_x = x_chain[t-1]
        current_y = y_chain[t-lag-1] #fingers crossed

        proposed_x, proposed_y = max_coupling_algo1(
            proposal_dist_logpdf(current_x), proposal_dist_logpdf(current_y),
            proposal_dist_sampler(current_x), proposal_dist_sampler(current_y)
        )

        log_u = log_unifs[t] # common random numbers

        if log_u <= log_alpha(current_x, proposed_x):
            x_chain[t] = proposed_x
        else:
            x_chain[t] = current_x

        if log_u <= log_alpha(current_y, proposed_y):
            y_chain[t-lag] = proposed_y
        else:
            y_chain[t-lag] = current_y

        if not meeting_time and y_chain[t-lag] == x_chain[t]:
            meeting_time = t
            logger.info(
                f"{meeting_time=} {x_chain[t]:.2f}"
            )

    y_chain[max_t_iterations-lag:] = np.nan

    #end timing now
    end_time = perf_counter()
    #record timing
    logger.info(
        f"chain took {round(end_time-start_time,3)} secs to simulate {max_t_iterations} iterations"
    )

    return DataFrame({"X":x_chain, "Y":y_chain})


def coupled_MCMC1( max_t_iterations=10**3):
    """simulate a normal with a hardcoded mean and sd"""

    #start timing here
    start_time = perf_counter()

    x_chain = np.zeros(max_t_iterations)
    y_chain = np.zeros(max_t_iterations)
    x_chain[0], y_chain[0] = -5, 5

    def log_alpha(current, new):
        """log of the alpha probability of accepting a proposed move.
        Here the proposal dist is symmetric and the target is N(3,4)        
        """
        mu = 3
        sigma_squared = 4   
        r = (current - new)*( current+new- 2*mu )/(2*sigma_squared)
        return min(0, r )

    #conscicously using common random numbers
    log_unif_rvs = np.log(uniform.rvs(size = max_t_iterations))
    for t in range(1, max_t_iterations):
        #propose a move from maximal coupling
        #here X_t+1 is sampled from Q(X_t,.), which is a normal centered on X_t with sd 1
        x_t_1, y_t_1 = x_chain[t-1], y_chain[t-1]

        proposed_x, proposed_y = max_coupling_algo1(
            norm(x_t_1, 1).logpdf, norm(y_t_1, 1).logpdf,
            norm(x_t_1, 1).rvs, norm(y_t_1, 1).rvs
        )
        
        #sample a uniform and take log
        log_u = log_unif_rvs[t]
        
        # logger.info(
        #     f"{log_u=:.2} {proposed_y=:.1f} {log_alpha(y_t_1, proposed_y)=:.2f}"
        #     +"\n\t\t"+
        #     f"{log_u=:.2} {proposed_x=:.1f} {log_alpha(x_t_1, proposed_x)=:.2f}"
        # )
        #decide if each chain accepts or rejects the move 
        if log_u <= log_alpha(x_t_1, proposed_x):
            x_chain[t] = proposed_x
        else:
            x_chain[t] = x_t_1
        
        if log_u <= log_alpha(y_t_1, proposed_y):
            y_chain[t] = proposed_y
        else:
            y_chain[t] = y_t_1

    #end timing now
    end_time = perf_counter()
    #record timing
    logger.info(
        f"chain took {round(end_time-start_time,3)} secs to simulate {max_t_iterations} iterations"
    )

    return DataFrame({"X":x_chain, "Y":y_chain})


def max_coupling_algo1(log_p_pdf, log_q_pdf, p_sampler, q_sampler, rng):
    """
    Sampling from a maximal coupling of x ~ p and y ~ q
    , using Chp3 Algorithm 1 from P.Jacob 2021

    requires a random number generator `rng`   
    """
    new_X = p_sampler(random_state = rng)
    u  = uniform.rvs(random_state = rng)
    if np.log(u) + log_p_pdf(new_X) <= log_q_pdf(new_X):
        return (new_X, new_X) # X=Y
    
    new_Y = None
    while new_Y is None: # when new_Y is N-D, a NOT would error
        proposed_Y = q_sampler(random_state = rng)
        u  = uniform.rvs(random_state = rng)
        if np.log(u) + log_q_pdf(proposed_Y) > log_p_pdf(proposed_Y):
            new_Y = proposed_Y

    return (new_X, new_Y)


def just_plot_it(x, y,  title = None ):

    fig, ax = plt.subplots()

    ax.scatter(x,y)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if title:
        plt.suptitle(title)
    plt.show()

def plot_joint_marginal(df, x_label = "X", y_label = "Y"):
    z = sns.JointGrid(df, x = x_label, y = y_label)
    z.plot_joint(sns.scatterplot)
    z.plot_marginals(sns.histplot, kde = True, color = "r")
    plt.show()

def trace_plot(sample, lag, meeting_time = None):

    fig, (ax1, ax2) = plt.subplots(ncols= 2)
    
    shifted_y = np.roll(sample["Y"], lag) # move the NAs from the end to the start
    ax2.plot(sample["X"],  "r")
    ax2.plot(shifted_y,  "b")
    ax2.set_xlabel("t + lag")

    if meeting_time:
        ax2.axvline(x = meeting_time, ls = "dashed", color = "blue")
    
    ax1.plot(sample["X"],  "r")
    ax1.plot(sample["Y"],  "b")
    ax1.set_xlabel("t")

    plt.show()

def trace_plot2(x:np.array,y:np.array,lag,  meeting_time = None):

    fig, ax1 = plt.subplots()

    ax1.set_title(f"Coupled chains with a lag of {lag}")

    y_shifted = np.pad(y,(lag,0), mode = "constant", constant_values = np.nan)
    ax1.plot(x,  "r")
    ax1.plot(y_shifted,  "b")
    ax1.set_xlabel("time")

    if meeting_time:
        ax1.axvline(x = meeting_time, ls = "dashed", color = "blue")
    
    ax1.legend(labels = ["Un-lagged","lagged"])


    plt.show()


def just_get_mvn_mcmc_chain( lag = 500, max_t_iterations=10**4, random_state = None):
    """
    A quick effort to get the actual chain generated 

    
    Takes a `random_state` for reproducability
    """
    mu  = np.array([3,4,5])
    P = len(mu)
    sigma = make_cov_equivar(P,.9,10) # high covariance

    sigma_inv = np.linalg.inv(sigma)

    #initialisation
    x_chain = np.zeros((max_t_iterations,P))
    y_chain = np.zeros((max_t_iterations,P))
    rng = np.random.default_rng(random_state)  

    #mu=0, sd =50 so this is a wide range of starting points
    x_chain[0], y_chain[0] = multivariate_normal.rvs(size =2  ,cov = (50**2)*np.identity(P), random_state =rng )
    #theres one spare here just to keep indexing simple
    log_unifs = np.log(uniform.rvs(size = max_t_iterations+1, random_state = rng)) 
    
    """
    Use a MVN proposal dist
      centred at the current for symmetry
      variance is probelem dependent, using 2 units in each direction
      setting covariances to zero so i dont have to worry about them
    """
    #abstraction
    def proposal_dist_logpdf(current_state):
        return multivariate_normal(mean = current_state, cov = (2**2)*np.identity(P)).logpdf 
    def proposal_dist_sampler(current_state):
        return multivariate_normal(mean = current_state, cov = (2**2)*np.identity(P)).rvs
    # def log_unnormalised_target_pdf(x): #not needed due to manual simplification of alpha
    #     pass
    
    
    def log_alpha(current, new):
        """simplified log of alpha using the function local `mu` and `sigma_inv`"""
        quad_new = quad_form_mvn(mu, sigma_inv, new)
        quad_old = quad_form_mvn(mu, sigma_inv, current)

        return min(0, -.5*(quad_new- quad_old))

    # run X chain for lag steps
    for t in range(1,lag+1):
        current_state = x_chain[t-1,]
        proposed_state = proposal_dist_sampler(current_state)(random_state =rng) # looks ugly i know

        if log_unifs[t] <= log_alpha(current_state, proposed_state):
            x_chain[t,] = proposed_state 
        else:
            x_chain[t,] = current_state 
    
    meeting_time = None
    # now run a coupling with the lagged chains
    for t in range(lag+1, max_t_iterations):
        current_x = x_chain[t-1,]
        current_y = y_chain[t-lag-1,] #fingers crossed 
        
        proposed_x, proposed_y = max_coupling_algo1(
            proposal_dist_logpdf(current_x), proposal_dist_logpdf(current_y),
            proposal_dist_sampler(current_x), proposal_dist_sampler(current_y)
            ,rng
        )

        log_u = log_unifs[t] # common random numbers

        if log_u <= log_alpha(current_x, proposed_x):
            x_chain[t,] = proposed_x
        else:
            x_chain[t,] = current_x

        if log_u <= log_alpha(current_y, proposed_y):
            y_chain[t-lag,] = proposed_y
        else:
            y_chain[t-lag,] = current_y

        if not meeting_time and (y_chain[t-lag,] == x_chain[t,]).all() : 
            # #first time meeting
            meeting_time = t
            # break # no need to continue, tau observed
            print(meeting_time)

    # if meeting_time is None:
    #     logger.warning(f"Chains did not meet after {max_t_iterations:,} steps {random_state=}")

    return x_chain, y_chain[0:(max_t_iterations-lag),]

def read_demo_df_file(f_name, f_subfolder):
    """convnience function (again) for reading files in the `good samples` folder"""
    folder = os.path.join("keep_safe", "good samples", f_subfolder)
    #inside df it will try join f_path to logs_and_data and fail, using just f_path
    return read_df_file(f_name, folder)

def plot_tau_stuff(tau_f, title_rv_name= ""):
    d = tau_f

    fig , (ax1) = plt.subplots(1)
    for c in d.columns:
        xs = d[c] -int(c) #tau - lag
        ax1.ecdf(xs, complementary  = True )

    ax1.set_title(f"ECCDF of tau for {title_rv_name}")
    ax1.set_xlabel("tau - lag")
    ax1.legend(d.columns, title = "lag")
    ax1.set_yscale("log")
    plt.show()

    # #show them indivivdually
    # for col in d.columns:
    #     fig , ax1 = plt.subplots(1)
    #     xs = d[col] -int(col) #tau - lag
    #     ax1.ecdf(xs, complementary  = True )

    #     ax1.set_title(f"ECCDF of tau when lag is {col}")
    #     ax1.set_xlabel("tau - lag")
    #     ax1.set_yscale("log")
    #     plt.show()

    #examine tau
    shifted = d.apply(lambda s:s -int(s.name), axis = 0)
    # print_basic_df_summary(shifted)
    fig , (ax1, ax2) = plt.subplots(1,2)
    fig.suptitle(title_rv_name)
    ax1.boxplot(d, tick_labels = d.columns)
    ax1.set_title("tau")
    ax2.boxplot(shifted, tick_labels = d.columns)
    ax2.set_title(f"tau-lag")
    plt.show(	)

def plot_tv_upper_bound(tv_est, title_rv_name=""):
    fig , ax1 = plt.subplots(1)
    ax1.plot(tv_est)
    ax1.axhline(0,color = "black", ls ="--")
    ax1.axhline(1,color = "black", ls ="--")
    ax1.axhline(0.25,color = "grey", ls ="--")
    ax1.set_title(f"TV upper bound for {title_rv_name}")
    ax1.legend(tv_est.columns, title = "Lag")
    ax1.set_ylabel("TV upper bound")
    ax1.set_xlabel("time t")
    plt.show()

def read_good_sample_np_csv(f_name, folder):
    f_path = os.path.join("keep_safe","good samples",folder
                          ,f_name)
    return np.genfromtxt(f_path, delimiter = ",")

def trace_plot_10_comps(chain):
    fig, axes = plt.subplots(5,2)
    axes = axes.flatten() 
    for i in range(chain.shape[1]):
        ax = axes[i]
        dat = chain[:,i]
        ax.plot(dat,"g-")
        ax.set_title(f"comp {i}")
    plt.show()

def boxplot_two_chains_side_by_side(a,b, a_name=None, b_name = None, var_names= None):
    """

    
    taken from https://stackoverflow.com/questions/43612687/python-matplotlib-box-plot-two-data-sets-side-by-side
    """
    fig, ax = plt.subplots()
    
    def draw_plot(data, offset,edge_color, fill_color, name):
        pos = np.arange(data.shape[1])+offset
        bp = ax.boxplot(data, positions= pos, widths=0.3, patch_artist=True, manage_ticks=False, sym ="", label = name)
        for element in ['boxes', 'whiskers', 'medians', 'caps']:
            plt.setp(bp[element], color=edge_color)
        for patch in bp['boxes']:
            patch.set(facecolor=fill_color)

    draw_plot(a, -0.2, "darkviolet", "violet", a_name)
    draw_plot(b, +0.2,"darkorange", "bisque", b_name)
    ax.legend()
    if var_names:
        plt.xticks(range(0,len(var_names)), var_names)
    plt.show()