import numpy as np
from pandas import DataFrame, read_csv
import multiprocessing
from time import perf_counter
from scipy.stats import norm, uniform, multivariate_normal
from functions import max_coupling_algo1, pretty_print_seconds, quad_form_mvn,make_cov_haar,make_cov_equivar
import os, logging
log_path = os.path.join("logs_and_data", "MCMCcouplingSimulation.log")#os safe
logging.basicConfig(filename = log_path , level=logging.INFO)
logger = logging.getLogger(__name__)


def sample_tau_L_for_many_lags(mcmc_algo ,lags:iter, num_tau_samples:int
                               , max_t_iterations = 10**5, starting_random_seed:int= 10101010 ):
    """
    `WARNING` only run this function inside an `__name__ == "__main__"` block.  `WARNING`
    """
    start_time = perf_counter()
    df = DataFrame()
    random_gen = np.random.default_rng(starting_random_seed) # explicitly get a seed generator

    # take any available processes and spread apply the tasks to them
    with multiprocessing.Pool() as pool: #context manager for cleanup
        for lag in lags:
            logger.info(f"\t\t getting {num_tau_samples:,} of tau at lag {lag:,}")

            #simulation parameters / args to the function making the chain
            args = list(  (lag, max_t_iterations, seed ) 
                    for seed in random_gen.integers(10**6, size = num_tau_samples )    )
            
            #use many processes to execute chain on different parameters
            df[lag]  = pool.starmap(mcmc_algo , args)
        
    end_time = perf_counter()
    logger.info(
        f"\t\t getting {num_tau_samples*len(lags):,} of tau took {pretty_print_seconds(end_time-start_time)}"
    )

    return df

def modified_coupled_MCMC2(lag:int, max_t_iterations=10**3, random_state = None):
    """
    coupling with a lag
    target a normal with a hardcoded mean and sd
    
    returns the first meeting time tau
    takes a random_state for reproducability
    """

    #initialisation
    x_chain = np.zeros(max_t_iterations)
    y_chain = np.zeros(max_t_iterations)
    #mu=0, sd =100 so this is a very wide range of starting points
    x_chain[0], y_chain[0] = norm.rvs(size =2  ,scale = 100, random_state =random_state )
    np.random.seed(random_state)  
    log_unifs = np.log(uniform.rvs(size = max_t_iterations+1)) #theres one spare here just to keep indexing simple

    
    #abstraction
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
    
    def max_coupling_algo1(log_p_pdf, log_q_pdf, p_sampler, q_sampler):
        """
        Sampling from a maximal coupling of x ~ p and y ~ q
        , using Chp3 Algorithm 1 from P.Jacob 2021
        
        """
        new_X = p_sampler()
        u  = uniform.rvs()
        if np.log(u) + log_p_pdf(new_X) <= log_q_pdf(new_X):
            # logger.info(f"meeting {new_X:.2f}")
            return (new_X, new_X) # X=Y
        
        new_Y = None
        while not new_Y:
            proposed_Y = q_sampler()
            u  = uniform.rvs()
            if np.log(u) + log_q_pdf(proposed_Y) > log_p_pdf(proposed_Y):
                new_Y = proposed_Y

        return (new_X, new_Y)
    
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
            #first time meeting
            meeting_time = t
            break # no need to continue, tau observed


    return meeting_time


def mvn_2d_mcmc(lag:int, max_t_iterations=10**3, random_state = None):
    """
    coupling with a lag
    target a MV Normal2 with a hardcoded mean and covariance
    
    
    takes a random_state for reproducability
    """
    #start timing here

    P = 2; mu = [3,4]; sigma = [[4,1],[1,2]]; sigma_inv = np.linalg.inv(sigma)

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
            #first time meeting
            meeting_time = t
            break # no need to continue, tau observed

    if meeting_time is None:
        logger.warning(f"Chains did not meet after {max_t_iterations:,} steps {random_state=}")

    return meeting_time

def mvn_Pd_mcmc(mu, sigma, lag:int, max_t_iterations=10**4, random_state = None, return_chain = False):
    """
    returns tau, the first meeting time, from a coupling of 
    two Pd norms with identical `mu` and `sigma` specified with lag `lag`.
    
    Takes a `random_state` for reproducability
    """
    #start timing here

    P = len(mu); sigma_inv = np.linalg.inv(sigma)

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
            #first time meeting
            meeting_time = t

            if not return_chain:
                break # no need to continue, tau observed

    if meeting_time is None:
        logger.warning(f"Chains did not meet after {max_t_iterations:,} steps {random_state=}")

    if return_chain:
        if meeting_time:
            logger.info(f"{meeting_time=}")
        return x_chain, y_chain[0:(max_t_iterations-lag),]
    else:
        return meeting_time

class Some_random_Pd_mcmc:
    """
    This is basically a wrapper for creating a random MVN target in MCMC
    """
    def __init__(self, p, seed):
        rng = np.random.default_rng(seed)
        self._mu = rng.uniform(-10**2, 10**2, size = p)
        scale = rng.poisson(5, size=1)[0]
        cov_seed = rng.integers(0, 10**6, size = 1)[0]
        self._sigma = make_cov_haar(cov_seed, p, 1/scale)
        self._p = p
        self._init_seed = seed

        #to use Java terms, defining a __name__ so that the class has a similar interface
        #to a function
        #adding in its initialisation args, to distinguish it from other instances
        self.__name__ = f"{self.__class__.__name__}-P{p}-Seed{seed}"

    def __str__(self):
        text = self.__name__
        
        if self._p ==1:
            #get a bit more info
            mu = self._mu[0]
            sigma_squared = self._sigma[0][0]
            text += f"mu{mu}-var{sigma_squared}"
        
        return text

    def __call__(self,*args, **kwargs):
        return mvn_Pd_mcmc(
            self._mu, self._sigma,
            *args, **kwargs
        )
    
    def generate_mcmc_sample(self, *args, **kwargs):
        return mvn_Pd_mcmc(
            self._mu, self._sigma,
            *args, **kwargs
        )
   
class high_autocorrelated_mvn(Some_random_Pd_mcmc):
    """Want mcmc on a normal with high autocorrelation"""

    def __init__(self, p, seed):
        super().__init__(p, seed)
        #dont care about mu override Sigma
        scale = np.median(self._mu) #why not, makes the variance large
        self._sigma = make_cov_equivar(p, .9, 1/scale)

def at1_8schools_coupled_mcmc(lag:int, random_state, max_t_iterations=10**4, return_chain =False):
    """
    Single chain target joint posterior in 8 schools problem
    """
    ind_counts = []
    #set up
    p = 8+2 #group means plus mu plus tau
    meeting_time = None
    DATA = read_csv(os.path.join("keep_safe","8SchoolsData.txt"))

    rng = np.random.default_rng(random_state)
    log_unifs = np.log(uniform.rvs(size = max_t_iterations+1, random_state = rng)) 

    x_chain, y_chain = np.zeros(shape=(max_t_iterations,p)), np.zeros(shape=(max_t_iterations,p))
    #could do this based on estimates from data
    x_chain[0,] = multivariate_normal(cov = (50**2)*np.identity(p)).rvs( random_state =rng)
    x_chain[0,p-1] = uniform.rvs(loc =1, scale =30, random_state =rng)#make sure population variance is positive
    y_chain[0,] = multivariate_normal(cov = (50**2)*np.identity(p)).rvs( random_state =rng)
    y_chain[0,p-1] = uniform.rvs(loc =1, scale =30, random_state =rng)#make sure population variance is positive

    #handy to have
    def get_v_mu_and_mu_hat_given_tau(tau:float):

        total_precision = np.sum( 1/(tau**2 +DATA["sigma.j"]**2)  )

        average = np.sum(DATA["yBar.j"]/(tau**2 +DATA["sigma.j"]**2 ) )
        precision_weighted_average = average/total_precision

        return total_precision, precision_weighted_average

    def get_v_j_and_theta_j_hat_given_hypparams(mu, tau, index):
            sample_mean = DATA["yBar.j"][index]
            sigma_sq = DATA["sigma.j"][index]**2
            tau_sq = tau**2
            var = 1/( (1/sigma_sq)+(1/tau_sq) )
            theta_j = ( (sample_mean/sigma_sq)+(mu/tau_sq) )*var

            return var, theta_j

    def log_unnormalised_target2(state, index):
            mu, tau = state[8], state[9]
            v_mu, mu_hat = get_v_mu_and_mu_hat_given_tau(tau)

            #mu part Normal conditioned on tau
            #the exp bit of a normal
            if index ==8:
                return -.5*v_mu*((mu-mu_hat)**2)

            #tau part 
            if index ==9:
                if tau <= 0:
                    return -np.inf #log of zero
                else:
                    log_total_prec = np.log(v_mu)

                    big_sum = np.log(DATA["sigma.j"]**2 + tau**2) + (DATA["yBar.j"] - mu_hat)**2/(DATA["sigma.j"]**2 + tau**2)
                    big_sum = np.sum(big_sum)

                    return -.5*( log_total_prec + big_sum )

            # group parts Normal conditioned on mu, tau
            var, theta_j = get_v_j_and_theta_j_hat_given_hypparams(mu, tau, index)
            #the exp bit of a normal
            return -.5*(1/var)*((state[index]-theta_j)**2)


    def log_alpha(current, new, index):
        r = log_unnormalised_target2(new, index)-log_unnormalised_target2(current,index)
        return min(0, r)

    #abstraction
    #get the proposal distribution for each component
    def proposal_general(index, current_state):
    
        mu, tau = current_state[8], current_state[9]
        if index <=7:#group effects
            v_j, theta_j = get_v_j_and_theta_j_hat_given_hypparams(mu, tau, index)
            return norm(loc = theta_j, scale = v_j**.5)
        
        elif index == 8:#population mean
            total_precision, precision_weighted_average = get_v_mu_and_mu_hat_given_tau(tau)
            return norm(loc = precision_weighted_average, scale = total_precision**-.5)
        
        elif index == 9:#population variance
            return norm(loc = tau, scale =5)
        
    #then in the two other functions either get a sampler or log density
    def proposal_dist_logpdf(index, current_state):
        return proposal_general(index, current_state).logpdf
    def proposal_dist_sampler(index, current_state):
        return proposal_general(index, current_state).rvs

    #run X chain for lag steps
    for t in range(1,lag+1):
        current_state = x_chain[t-1,]

        proposed_state = current_state.copy()
        #update a randomly selected component
        index = rng.choice(p,1)[0]
        proposed_state[index] = proposal_dist_sampler(index, current_state)(random_state=rng)

        #accept/reject
        log_u = log_unifs[t]
        if log_u <= log_alpha(current_state, proposed_state, index):
            x_chain[t] = proposed_state
        else:
            x_chain[t] = current_state
    
    #couple the chains
    for t in range(lag+1, max_t_iterations):
        current_x, current_y = x_chain[t-1], y_chain[t-lag-1]

        proposed_x, proposed_y = current_x.copy(), current_y.copy()
        #update a randomly selected component
        index = rng.choice(p,1)[0]
        new_comp_x, new_comp_y = max_coupling_algo1(
                proposal_dist_logpdf(index, current_x)
                ,proposal_dist_logpdf(index, current_y)
                ,proposal_dist_sampler(index, current_x)
                ,proposal_dist_sampler(index, current_y)
                ,rng             
        )
        proposed_x[index], proposed_y[index] = new_comp_x, new_comp_y 
        
        #accept/reject with common random numbers
        log_u = log_unifs[t]

        if log_u <= log_alpha(current_x, proposed_x, index):
            x_chain[t] = proposed_x
        else:
            x_chain[t] = current_x
        if log_u <= log_alpha(current_y, proposed_y, index):
            y_chain[t-lag] = proposed_y
        else:
            y_chain[t-lag] = current_y

        #check for meeting
        if not meeting_time and (y_chain[t-lag,] == x_chain[t,]).all() : 
            #first time meeting
            meeting_time = t
            if not return_chain:
                break # no need to continue, tau observed

    if meeting_time is None:
        logger.warning(f"Chains did not meet after {max_t_iterations:,} steps {random_state=}")
    
    if return_chain:
        if meeting_time:
            logger.info(f"{meeting_time=}")
        #get rid of the initialised bits of Y never populated
        return x_chain, y_chain[0:(max_t_iterations-lag),]
    else:
        return meeting_time
    

if __name__ == "__main__":
    N= 10_000
    lag = N//10
    burn_in =N//5
    x, y = at1_8schools_coupled_mcmc(lag, 55, max_t_iterations =N ,  return_chain =True)
    from functions import trace_plot2
    for comp in range(x.shape[1]):
        trace_plot2(x[burn_in:,comp], y[burn_in:,comp], lag)
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.histplot(x[:,9], bins = 30)
    plt.show()