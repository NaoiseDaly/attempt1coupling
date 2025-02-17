from generate_tau_samples import *
from functions import make_cov_haar, make_cov_ar1
import os
import numpy as np
import logging

def make_log_path_str(f_name):
    """os safe. Save to logs and data so it can be easily ran and moved on remote"""
    return os.path.join("logs_and_data", f_name)

if __name__ == "__main__":
    
    #make a logger for this file
    log_path = make_log_path_str( "MCMCcouplingSimulation.log")
    logging.basicConfig(filename = log_path , level=logging.INFO)
    remote_logger = logging.getLogger(__name__)
    remote_logger.info("\n") # add a line to seperate this execution from any others
    
    #original mcmc algorithms
    for algo in [modified_coupled_MCMC2, mvn_2d_mcmc,
                 Some_random_Pd_mcmc(1, 9),Some_random_Pd_mcmc(2,10), Some_random_Pd_mcmc(3,11) ]:
        remote_logger.info(f"running {algo.__name__}")
        tau_data = sample_tau_L_for_many_lags(
            algo,
            lags = [300, 500, 800],
            num_tau_samples = 100,
            starting_random_seed =10101010
            )   
        
        f_name = f"check_reproducability_sample_tau_L_for_many_lags__{algo.__name__}.csv"
        f_path = make_log_path_str( f_name)

        tau_data.to_csv(f_path, index = True)

        remote_logger.info(f"saved to {f_path}")
    
    #covariance matrices
    p = 10; num_matrices = 10
    for func in [make_cov_haar, make_cov_ar1]: 

        cov_dat = np.zeros((num_matrices,p,p))
        for i, seed in enumerate(range(  33**3, num_matrices+ 33**3)):
            cov_dat[i, ] = func(seed, p, i+1)

        f_name = f"reproduce__{func.__name__}.npy"
        f_path = make_log_path_str( f_name)
        np.save(f_path, cov_dat)
        remote_logger.info(f"saved to {f_path}")
    
    remote_logger.info("Done \n")
