from functions import *
from generate_tau_samples import sample_tau_L_for_many_lags

logger = logging.getLogger(__name__)
logging.basicConfig( level=logging.INFO)

results_df = sample_tau_L_for_many_lags(
    lags = [100*i for i in range(3,9)], 
    num_tau_samples = 100)
print_basic_df_summary(results_df)

f = save_df_with_timestamp(results_df, "tau lag")

logger.info(f"saved TV ests as {f}")