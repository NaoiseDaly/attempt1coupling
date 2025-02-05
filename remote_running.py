from functions import *
from generate_tau_samples import sample_tau_L_for_many_lags

logger = logging.getLogger(__name__)
log_path = os.path.join("logs_and_data", "MCMCcouplingSimulation.log")#os safe
logging.basicConfig(filename = log_path , level=logging.INFO)

tau_data = sample_tau_L_for_many_lags(
    lags = [300, 500, 800], 
    num_tau_samples = 10_000)
print_basic_df_summary(tau_data)

#save just in case
tau_data_f = save_df_with_timestamp(tau_data, "tau lag")
logger.info(f"saved tau samples as {tau_data_f}")

tv_data_f = estimate_TV_from_file(tau_data_f, num_ts = 500)
logger.info(f"saved TV ests as {tv_data_f}")

logger.info("\n\n") # to seperate entries