import numpy as np
from pandas import DataFrame
import multiprocessing
from time import perf_counter
from scipy.stats import norm, uniform, multivariate_normal
from functions import max_coupling_algo1, pretty_print_seconds, quad_form_mvn
import os, logging
log_path = os.path.join("logs_and_data", "MCMCcouplingSimulation.log")#os safe
logging.basicConfig(filename = log_path , level=logging.INFO)
logger = logging.getLogger(__name__)


def sample_tau_L_for_many_lags(mcmc_algo ,lags:iter, num_tau_samples:int
                               , max_t_iterations = 10**5, starting_random_seed:int= 10101010 ):
    
    start_time = perf_counter()
    df = DataFrame()
    random_gen = np.random.default_rng(starting_random_seed) # explicitly get a seed generator
    # take any available processes and spread apply the tasks to them
    with multiprocessing.Pool() as pool: #context manager for cleanup
        for lag in lags:
            logger.info(f"\t\t getting {num_tau_samples:,} of tau at lag {lag:,}")
            #simulation parameters / args to the function making the chain
            args = [ (lag, max_t_iterations, seed ) 
                    for seed in random_gen.integers(10**6, size = num_tau_samples )    ]
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
        #start timing here
    start_time = perf_counter()

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

    #end timing now
    end_time = perf_counter()
    #record timing
    # logger.info(
    #     f"{random_state=} \t {round(end_time-start_time,1)} secs  {t} iterations, tau {meeting_time}"
    # )

    return meeting_time

def mcmc3(lag:int, max_t_iterations=10**3, random_state = None):
    """
    coupling with a lag
    target a MV Normal2 with a hardcoded mean and covariance
    
    
    takes a random_state for reproducability
    """
        #start timing here
    start_time = perf_counter()

    P = 2; mu = None

    #initialisation
    x_chain = np.zeros(max_t_iterations)
    y_chain = np.zeros(max_t_iterations)
    rng = np.random.default_rng(random_state)  

    #mu=0, sd =100 so this is a very wide range of starting points
    x_chain[0], y_chain[0] = norm.rvs(size =2  ,scale = 100, random_state =rng )
    #theres one spare here just to keep indexing simple
    log_unifs = np.log(uniform.rvs(size = max_t_iterations+1, random_state = rng)) 
    
    #abstraction
    def proposal_dist_logpdf(current_state):
        return norm(current_state, 1).logpdf
    def proposal_dist_sampler(current_state):
        return norm(current_state, 1).rvs
    def log_unnormalised_target_pdf(x):
        return norm(loc = 3,scale  = 2).logpdf(x)
    
    
    def log_alpha(current, new):
        top = log_unnormalised_target_pdf(new) + proposal_dist_logpdf(new)(current)
        bottom = log_unnormalised_target_pdf(current) + proposal_dist_logpdf(current)(new)
        return min( 0, top - bottom )

    def max_coupling_algo1(log_p_pdf, log_q_pdf, p_sampler, q_sampler):
        """
        Sampling from a maximal coupling of x ~ p and y ~ q
        , using Chp3 Algorithm 1 from P.Jacob 2021
        
        """
        new_X = p_sampler(random_state = rng)
        u  = uniform.rvs(random_state = rng)
        if np.log(u) + log_p_pdf(new_X) <= log_q_pdf(new_X):
            # logger.info(f"meeting {new_X:.2f}")
            return (new_X, new_X) # X=Y
        
        new_Y = None
        while not new_Y:
            proposed_Y = q_sampler(random_state = rng)
            u  = uniform.rvs(random_state = rng)
            if np.log(u) + log_q_pdf(proposed_Y) > log_p_pdf(proposed_Y):
                new_Y = proposed_Y

        return (new_X, new_Y)


    # run X chain for lag steps
    for t in range(1,lag+1):
        current_state = x_chain[t-1]
        proposed_state = proposal_dist_sampler(current_state)(random_state =rng) # looks ugly i know

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
            print(meeting_time)
            # break # no need to continue, tau observed

    if meeting_time is None:
        logger.warning(f"Chains did not meet after {max_t_iterations:,} steps {random_state=}")

    return DataFrame(dict({"X":x_chain, "Y":y_chain}))

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
    x_chain[0], y_chain[0] = multivariate_normal.rvs(size =2  ,cov = 250*np.identity(P), random_state =rng )
    #theres one spare here just to keep indexing simple
    log_unifs = np.log(uniform.rvs(size = max_t_iterations+1, random_state = rng)) 
    
    """
    Use a MVN proposal dist
      centred at the current for symmetry
      variance is probelem dependent, using 5 units in each direction
      setting covariances to zero so i dont have to worry about them
    """
    #abstraction
    def proposal_dist_logpdf(current_state):
        return multivariate_normal(mean = current_state, cov = 25*np.identity(P)).logpdf 
    def proposal_dist_sampler(current_state):
        return multivariate_normal(mean = current_state, cov = 25*np.identity(P)).rvs
    # def log_unnormalised_target_pdf(x): #not needed due to manual simplification of alpha
    #     pass
    
    
    def log_alpha(current, new):
        """simplified log of alpha using the function local `mu` and `sigma_inv`"""
        quad_new = quad_form_mvn(mu, sigma_inv, new)
        quad_old = quad_form_mvn(mu, sigma_inv, current)

        return min(0, -.5*(quad_new- quad_old))

    def max_coupling_algo1(log_p_pdf, log_q_pdf, p_sampler, q_sampler):
        """
        Sampling from a maximal coupling of x ~ p and y ~ q
        , using Chp3 Algorithm 1 from P.Jacob 2021

        uses a function-local random number generator `rng`   
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