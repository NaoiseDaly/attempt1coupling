from functions import print_basic_df_summary, save_df_with_timestamp, estimate_TV_from_file, make_timestamp,read_demo_df_file
from generate_tau_samples import *
from unit_tests import run_all_checks
import logging, os.path
#make a logger for this file
log_path = os.path.join("logs_and_data", "MCMCcouplingSimulation.log")#os safe
logging.basicConfig(filename = log_path , level=logging.INFO)
remote_logger = logging.getLogger(__name__)

def get_tv_est_8schools():

    tau_data = sample_tau_L_for_many_lags(
        at1_8schools_coupled_mcmc
        ,lags = [1_000, 2_000, 3_000]
        , num_tau_samples = 1_000
    )
    print_basic_df_summary(tau_data)

    f_name = save_df_with_timestamp(tau_data, f"{at1_8schools_coupled_mcmc.__name__}-tau-data")
    remote_logger.info(f"tau samples saved to {f_name}")

    f_name = estimate_TV_from_file(f_name, 600, f"{at1_8schools_coupled_mcmc.__name__}-tv-ests")
    remote_logger.info(f"Tv estimates saved to {f_name}")

def run_2_chain_8schools():

    tv_bound = read_demo_df_file("at1_8schools_coupled_mcmc-tv-ests 2025-03-15 Sat 21-54.csv"
                       ,"8schools example" )["3000"]

    #the TV is non increasing so this is safe
    t_short = tv_bound[tv_bound <=.25].first_valid_index()
    t_long = tv_bound[tv_bound <=(1-.99)].first_valid_index()
    chain_size = 2_000
    chain_factor = 10

    remote_logger.info(f"getting two chains of lengths {t_short}, {t_long}")
    #because of the implementations, I know that evene with the same seed
    #  because the sizes are different this gives distinct chains
    #(the uniforms are precomputed at the start so the RNGs would go out of sync)
    long, _ = at1_8schools_coupled_mcmc(
        lag = 1, 
        random_state = 505
        , return_chain = True
        ,max_t_iterations = t_long*chain_factor
    )
    short, _ = at1_8schools_coupled_mcmc(
        lag = 1, 
        random_state = 505
        , return_chain = True
        ,max_t_iterations = t_short*chain_factor
    )
    remote_logger.info("writing chains to file")

    stamp = make_timestamp()
    long_path = os.path.join("logs_and_data", f"8schools_long_chain_{stamp}.csv" )
    short_path = os.path.join("logs_and_data", f"8schools_short_chain_{stamp}.csv" )
    np.savetxt(fname = long_path,X =  long[t_long:], delimiter = ",")
    np.savetxt(fname = short_path,X =  short[t_short:], delimiter = ",")

    remote_logger.info("done now")

def get_big_mcmc_sample():
    
    mvn  = high_autocorrelated_mvn(3, 42)
    tau_data = sample_tau_L_for_many_lags(
        mvn,
        lags = [300, 500, 800, 1100],
        num_tau_samples = 10_000)
    print_basic_df_summary(tau_data)
    f_name = save_df_with_timestamp(tau_data, f"{mvn.__name__}-tau-data")
    remote_logger.info(f"tau samples saved to {f_name}")


    quantiles = tau_data.quantile(q = .99)
    #get the median of the burnins for each lag
    #center tau by the lag
    burn_in =   np.median(
        [ quantiles[lag] -int(lag) for lag in quantiles.index ]
    )
    burn_in  = int(np.ceil(burn_in)) # need it to be a whole number
    remote_logger.info(f"using {burn_in=}")


    #in case this is comp heavy id like to do it on the remote
    # setting t in the range 1.5 * burn in is arbitarty
    f_name = estimate_TV_from_file(f_name, int(burn_in*1.5), f"{mvn.__name__}-tv-ests")
    remote_logger.info(f"Tv estimates saved to {f_name}")


    #this is just one chain so cant be parrallelised
    remote_logger.info(f"running a {burn_in}lagged coupled chain for {10*burn_in:,} steps")
    x_chain, y_chain = mvn.generate_mcmc_sample(
        lag = burn_in, return_chain = True
        , max_t_iterations = 10*burn_in
        ,random_state = 500_0_005
        )

    stamp = make_timestamp()
    y_path = os.path.join("logs_and_data", f"{mvn.__name__}_lagged_chain_{stamp}.csv" )
    x_path = os.path.join("logs_and_data", f"{mvn.__name__}_unlagged_chain{stamp}.csv" )
    np.savetxt(fname = y_path,X =  y_chain, delimiter = ",")
    np.savetxt(fname = x_path,X =  x_chain, delimiter = ",")

    remote_logger.info(f"final chain wrote to file")


if __name__ == "__main__":

    # run_all_checks()


    remote_logger.info("\n") # add a line to seperate this execution from any others


    # get_tv_est_8schools()
    run_2_chain_8schools()

    # get_big_mcmc_sample()
    # # do not call sample_tau_L_for_many_lags outside of here 
    # tau_data = sample_tau_L_for_many_lags(
    #     Some_random_Pd_mcmc(p = 3,seed = 5),
    #     lags = [300, 500, 800],
    #     num_tau_samples = 10_000)
    # print_basic_df_summary(tau_data)

    # #save just in case
    # tau_data_f = save_df_with_timestamp(tau_data, "tau lag")
    # remote_logger.info(f"saved tau samples as {tau_data_f}")

    # tv_data_f = estimate_TV_from_file(tau_data_f, num_ts = 500)
    # remote_logger.info(f"saved TV ests as {tv_data_f}")

    remote_logger.info("\n") # to seperate entries