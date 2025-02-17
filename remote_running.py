from functions import print_basic_df_summary, save_df_with_timestamp, estimate_TV_from_file
from generate_tau_samples import *
from unit_tests import     run_all_checks
import logging, os.path

if __name__ == "__main__":

    run_all_checks()

    #make a logger for this file
    log_path = os.path.join("logs_and_data", "MCMCcouplingSimulation.log")#os safe
    logging.basicConfig(filename = log_path , level=logging.INFO)
    remote_logger = logging.getLogger(__name__)
    remote_logger.info("\n") # add a line to seperate this execution from any others

    # do not call sample_tau_L_for_many_lags outside of here 
    tau_data = sample_tau_L_for_many_lags(
        Some_random_Pd_mcmc(p = 3,seed = 5),
        lags = [300, 500, 800],
        num_tau_samples = 1_000)
    print_basic_df_summary(tau_data)

    #save just in case
    tau_data_f = save_df_with_timestamp(tau_data, "tau lag")
    remote_logger.info(f"saved tau samples as {tau_data_f}")

    tv_data_f = estimate_TV_from_file(tau_data_f, num_ts = 500)
    remote_logger.info(f"saved TV ests as {tv_data_f}")

    remote_logger.info("\n") # to seperate entries