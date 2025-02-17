from functions import *
import os, logging
from generate_tau_samples import *
import numpy as np
from numpy.random import default_rng
log_path = os.path.join("logs_and_data", "MCMCcouplingSimulation.log")#os safe
logging.basicConfig(filename = log_path , level=logging.INFO)
test_logger = logging.getLogger(__name__)

FOLDER_PATH = os.path.join("keep_safe", "unit_test_data")



def reproduce__sample_tau_L_for_many_lags(algo):
    """
    check reproducability by running `algo` with hardcoded inputs and seed, and 
    then comparing to prior saved output of tau.

    For simplicity every `algo` has been executed on the same lags and seed
    """
    #run the simulation now using `starting_random_seed` = 10101010
    current_answer = sample_tau_L_for_many_lags(
        algo,
        lags = [300, 500, 800], 
        num_tau_samples = 100, starting_random_seed= 10101010)

    #load result carried out before using `starting_random_seed` = 10101010
    f_name = f"check_reproducability_sample_tau_L_for_many_lags__{algo.__name__}.csv"

    original_answer = read_df_file(f_name, FOLDER_PATH)
    # nice little catch that would make a df comparison fail
    original_answer.columns = [ int(c) for c in original_answer.columns ] 

    # compare dfs
    assert  original_answer.equals(current_answer) 

def reproduce__make_cov_funcs(func):
    """
    check the reproducability of `func` by running it 
    with hardcoded inputs and seed, and then comparing to prior saved output

    `func` should be one of the following:
    ``
    """
    p = 10; num_matrices = 10

    current_answer = np.zeros((num_matrices,p,p))
    for i, seed in enumerate(range(  33**3, num_matrices+ 33**3)):
        current_answer[i, ] = func(seed, p, i+1)

    f_name = f"reproduce__{func.__name__}.npy"
    f_path = os.path.join(FOLDER_PATH, f_name)
    original_answer = np.load(f_path)

    assert np.array_equal(original_answer, current_answer)

def safe_test(test, func):
    try:
        test(func)
    except AssertionError as e:
        msg = f"\t\t{func.__name__} failed unit test {test.__name__} "
        test_logger.error(msg)
        print(msg)
    


def run_all_checks():
    """
    Run all units tests, only reports failures.
    Side-effects may include excessive logging
    
    Please ensure this is nested somehow in an `if __name__ = "__main__"`"""

    test_logger.info("\t starting unit tests")

    for test, test_args in [
         (reproduce__sample_tau_L_for_many_lags, 
          [modified_coupled_MCMC2, mvn_2d_mcmc,
           Some_random_Pd_mcmc(1, 9),Some_random_Pd_mcmc(2,10), Some_random_Pd_mcmc(3,11)])
        ,(reproduce__make_cov_funcs, [make_cov_haar, make_cov_ar1])
        ]:
        for func in test_args:
            safe_test(test, func)
        




    test_logger.info("\t finished unit tests \n")

if __name__ == "__main__":
    run_all_checks()