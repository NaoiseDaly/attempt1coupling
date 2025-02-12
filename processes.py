import multiprocessing 
import time
from random import Random
import numpy as np
import os
from pandas import DataFrame
from random import Random
from time import perf_counter
from scipy.stats import norm, uniform
from functions import max_coupling_algo1, pretty_print_seconds
from generate_tau_samples import sample_tau_L_for_many_lags
import logging
logger = logging.getLogger(__name__)
logging.basicConfig( level=logging.INFO)

from generate_tau_samples import modified_coupled_MCMC2

def f(a,b,c):
    time.sleep(5)
    return f"{a}-{b}-{c}"

def sample_tau_L_for_many_lags2(lags:iter, num_tau_samples  =5, max_t_iterations = 10**5, starting_random_seed:int= 10101010):
    """Tries to parrallelise things"""
    start_time = perf_counter()
    df = DataFrame()
    random_gen = np.random.default_rng(starting_random_seed) # explicitly get a seed generator

    for lag in lags:
        logger.info(f"\t\t getting {num_tau_samples:,} of tau at lag {lag:,}")
        #simulation parameters / args to the function making the chain
        args = [ (lag, max_t_iterations, seed ) for seed in random_gen.integers(0,10**6, size = num_tau_samples )    ]
        
        # take any available processes and spread apply the tasks to them
        with multiprocessing.Pool() as pool: #context manager for cleanup
            arr = pool.starmap(modified_coupled_MCMC2 , args)
        df[lag] = arr
    
    end_time = perf_counter()
    logger.info(
        f"\t\t getting {num_tau_samples*len(lags):,} of tau took {pretty_print_seconds(end_time-start_time)}"
    )

    return df


if __name__ == "__main__":
    # pool = multiprocessing.Pool(5)
    # s = time.perf_counter()

    # rand_gen = Random(1)

    # params = [ (lag,rand_gen.randint(0, 10^5),7) for lag in inputs for _ in range(10, 50+1, 10)]
    # # for pair in params:
    # #     print(pair)
    # outputs = pool.starmap(f, params)


    # e = time.perf_counter()
    # print( outputs, int(e-s) )
    # s = time.perf_counter()
    # outputs = [ f(i) for i in inputs]
    # e = time.perf_counter()
    # print( outputs, int(e-s) )
    
    inputs = [300, 500, 800]; NUM_TAU =120
    out = sample_tau_L_for_many_lags2(inputs,NUM_TAU , starting_random_seed=  1)
    s = time.perf_counter()
    sample_tau_L_for_many_lags(inputs, NUM_TAU, starting_random_seed=  1)
    e = time.perf_counter()
    print("orig", int(e-s))


   