from functions import read_df_file
import os, logging
from generate_tau_samples import *
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


def run_all_checks():
    """
    Run all units tests, only reports failures.
    Side-effects may include excessive logging
    
    Please ensure this is nested somehow in an `if __name__ = "__main__"`"""
    test_logger.info("\t starting unit tests")
    tests = [
        reproduce__sample_tau_L_for_many_lags
        ]
    funcs = [modified_coupled_MCMC2, mvn_2d_mcmc]
    for test in tests:
        for func in funcs:
            try:
                test(func)
            except AssertionError as e:
                msg = f"{func.__name__} failed unit test {test.__name__} "
                test_logger.error(msg)
                print(msg)

    test_logger.info("\t finished unit tests \n")

if __name__ == "__main__":
    run_all_checks()