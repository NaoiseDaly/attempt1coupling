from functions import *
from generate_tau_samples import sample_tau_L_for_many_lags

FOLDER_PATH = os.path.join("keep_safe", "unit_test_data")



def check_reproducability_sample_tau_L_for_many_lags():

    #load result carried out before using `starting_random_seed` = 1
    f_name = "check_reproducability_sample_tau_L_for_many_lags1.csv"

    original_answer = read_df_file(f_name, FOLDER_PATH)
    # nice little catch that would make a df comparison fail
    original_answer.columns = [ int(c) for c in original_answer.columns ] 

    #run the simulation now using `starting_random_seed` = 1
    current_answer = sample_tau_L_for_many_lags(
        lags = [300, 500, 800], 
        num_tau_samples = 5, starting_random_seed=1)
    
    # compare dfs
    assert  original_answer.equals(current_answer) 



#run all tests
check_reproducability_sample_tau_L_for_many_lags()