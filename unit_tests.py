from functions import read_df_file
import os, logging
from generate_tau_samples import sample_tau_L_for_many_lags, mcmc3
log_path = os.path.join("logs_and_data", "MCMCcouplingSimulation.log")#os safe
logging.basicConfig(filename = log_path , level=logging.INFO)
test_logger = logging.getLogger(__name__)

FOLDER_PATH = os.path.join("keep_safe", "unit_test_data")



def check_reproducability_sample_tau_L_for_many_lags():

    #load result carried out before using `starting_random_seed` = 10101010
    f_name = "check_reproducability_sample_tau_L_for_many_lags2.csv"

    original_answer = read_df_file(f_name, FOLDER_PATH)
    # nice little catch that would make a df comparison fail
    original_answer.columns = [ int(c) for c in original_answer.columns ] 

    #run the simulation now using `starting_random_seed` = 10101010
    current_answer = sample_tau_L_for_many_lags(
        mcmc3,
        lags = [300, 500, 800], 
        num_tau_samples = 100, starting_random_seed= 10101010)
    
    # compare dfs
    assert  original_answer.equals(current_answer) 


def run_all_checks():
    """
    Run all units tests, only reports failures.
    Side-effects may include excessive logging
    
    Please ensure this is nested somehow in an `if __name__ = "__main__"`"""
    test_logger.info("\t starting unit tests")
    for test in [
        check_reproducability_sample_tau_L_for_many_lags
        ]:
        try:
            test()
        except AssertionError as e:
            msg = f"unit test {test.__name__} failed"
            test_logger.error(msg)
            print(msg)

    test_logger.info("\t finished unit tests \n")

if __name__ == "__main__":
    run_all_checks()