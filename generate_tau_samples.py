import numpy as np
from pandas import DataFrame
from time import perf_counter
from scipy.stats import norm, uniform
from functions import max_coupling_algo1
import logging
logger = logging.getLogger(__name__)



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
    x_chain[0], y_chain[0] = -100, 100   #may need to review this
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

    y_chain[max_t_iterations-lag:] = np.nan # leave the rest blank

    #end timing now
    end_time = perf_counter()
    #record timing
    logger.info(
        f"{random_state=} chain took {round(end_time-start_time,3)} secs to simulate {max_t_iterations} iterations, tau {meeting_time}"
    )

    return meeting_time, DataFrame({"X":x_chain, "Y":y_chain})