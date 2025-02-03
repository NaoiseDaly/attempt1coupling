import numpy as np
from pandas import DataFrame
from time import perf_counter
from scipy.stats import norm, uniform
from functions import max_coupling_algo1
import logging
logger = logging.getLogger(__name__)


def sample_tau_L_for_many_lags(lags:iter, num_tau_samples  =5, max_t_iterations = 10**5, starting_random_seed:int= 10101010 ):
    
    start_time = perf_counter()

    df = DataFrame()
 
    for l in lags:
        logger.info(f"\t\t getting {num_tau_samples} of tau at lag {l}")
        arr = np.zeros( ( num_tau_samples))
        for i, seed in enumerate( range(starting_random_seed, starting_random_seed + num_tau_samples)  ):
            arr[i] = modified_coupled_MCMC2(l, max_t_iterations, seed) 
        df[l] = arr
    
    end_time = perf_counter()
    logger.info(
        f"\t\t getting {num_tau_samples*len(lags):,} of tau took {round(end_time-start_time,1)} secs  "
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
    #mu=0, sd =50 so this is a very wide range of starting points
    x_chain[0], y_chain[0] = norm.rvs(size =2  ,scale = 50, random_state =random_state )
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