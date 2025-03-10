from functions import *
import logging 
logger = logging.getLogger(__name__)
logging.basicConfig( level=logging.INFO)
np.random.seed(1345301)

"""looking at the maximal coupling algorithm on its own"""
# sample = [
#     max_coupling_algo1(norm(5,3).logpdf, norm(10,1).logpdf, norm(5,3).rvs, norm(10,1).rvs,5)
#     for _ in range(200)]
# df = pd.DataFrame({"X":[pair[0] for pair in sample],
#                    "Y":[ pair[1] for pair in sample]})


# just_plot_it(df["X"], df["Y"])
# plot_joint_marginal(df)

"""Doing MCMC with the maximal coupling algorithm as the proposal dist"""
# N = 2000
# sample = coupled_MCMC1(N)
# sample_after_burn_in = sample.iloc[N//5:]

# print_basic_df_summary(sample)
# print_basic_df_summary(sample_after_burn_in)

# plot_joint_marginal(sample)
# plot_joint_marginal(sample_after_burn_in)
# fig, (ax1, ax2) = plt.subplots(nrows = 2)



# fig, ax1 = plt.subplots(nrows = 1)

# ax1.plot(sample["X"],  "r")
# ax1.plot(sample["Y"],  "b")
# plt.show()

"""Doing coupled MCMC with a lag"""
# N, L = 600, 200
# c_sample = coupled_MCMC2(lag = L,max_t_iterations= N)

# trace_plot(c_sample, L, 453)
# print_basic_df_summary(c_sample)
# print_basic_df_summary(c_sample.iloc[((N-L)//5)+L:])

def read_demo_df_file(f_name, f_subfolder):
    """convnience function (again) for reading files in the `good samples` folder"""
    folder = os.path.join("keep_safe", "good samples", f_subfolder)
    #inside df it will try join f_path to logs_and_data and fail, using just f_path
    return read_df_file(f_name, folder)

def plot_tau_stuff(tau_f, title_rv_name= ""):
    d = tau_f

    fig , (ax1) = plt.subplots(1)
    for c in d.columns:
        xs = d[c] -int(c) #tau - lag
        ax1.ecdf(xs, complementary  = True )

    ax1.set_title(f"ECCDF of tau for {title_rv_name}")
    ax1.set_xlabel("tau - lag")
    ax1.legend(d.columns, title = "lag")
    ax1.set_yscale("log")
    plt.show()

    # #show them indivivdually
    # for col in d.columns:
    #     fig , ax1 = plt.subplots(1)
    #     xs = d[col] -int(col) #tau - lag
    #     ax1.ecdf(xs, complementary  = True )

    #     ax1.set_title(f"ECCDF of tau when lag is {col}")
    #     ax1.set_xlabel("tau - lag")
    #     ax1.set_yscale("log")
    #     plt.show()

    #examine tau
    shifted = d.apply(lambda s:s -int(s.name), axis = 0)
    # print_basic_df_summary(shifted)
    fig , (ax1, ax2) = plt.subplots(1,2)
    fig.suptitle(title_rv_name)
    ax1.boxplot(d, tick_labels = d.columns)
    ax1.set_title("tau")
    ax2.boxplot(shifted, tick_labels = d.columns)
    ax2.set_title(f"tau-lag")
    plt.show(	)

def plot_tv_upper_bound(tv_est, title_rv_name=""):
    fig , ax1 = plt.subplots(1)
    ax1.plot(tv_est)
    ax1.axhline(0,color = "black", ls ="--")
    ax1.set_title(f"TV upper bound for {title_rv_name}")
    ax1.legend(tv_est.columns, title = "Lag")
    ax1.set_ylabel("TV upper bound")
    ax1.set_xlabel("time t")
    plt.show()

"""Sample a lot of tau and estimate the TV upper bound for the chains

tv est 2025-02-04 Tue 13-36.csv 
has a sample of 10K when chains start points are initialised
by N(0,50^2) 

tv est 2025-02-05 Wed 18-46.csv 
is also a sample of 10K but chains are initialised
by N(0, 100^2) - this is more interesting when compared to the previous. Tau obviously has higher sample variance 
- the number of steps before the chains meet (tua-lag) has mean ~200 and std error ~150, 
This and as seen from the boxplots the IQR is very close to 300 implying that the unlagged chain may not even have 
reached stationarity before coupling. This explains the quite vacous bound on the TV distance given by the lag 300 sample.
Also the bound was not effectively zero by 500 iterations for any lag either-whereas it was for the previous set up before.


"""
#plot TV upper bound
# demo_folder = "comp init N1"
# tv_f = read_demo_df_file("tv est 2025-02-04 Tue 13-36.csv", demo_folder)
# plot_tv_upper_bound(tv_f, "N(3,4) with tight initialisation")

# tau_f = read_demo_df_file("tau lag 2025-02-04 Tue 13-36.csv", demo_folder)
# plot_tau_stuff(tau_f, "N(3,4) with tight initialisation")

# tv_f = read_demo_df_file("tv est 2025-02-05 Wed 18-46.csv", demo_folder)
# plot_tv_upper_bound(tv_f, "N(3,4) with wider initialisation")

# tau_f = read_demo_df_file("tau lag 2025-02-05 Wed 18-46.csv", demo_folder)
# plot_tau_stuff(tau_f, "N(3,4) with wider initialisation")


"""
For multivariate norm
mu = [3,4]; sigma = [[4,1],[1,2]]

Compared to the univariate example above the chains converge quickly
The TV bound also appears almost sinusoidal
"""
# demo_folder = "N--3-4--4-1-1-2--10Ks"

# tau_f = read_demo_df_file("tau lag 2025-02-16 Sun 16-28.csv", demo_folder)
# tv_f = read_demo_df_file("TV est 2025-02-16 Sun 16-28.csv", demo_folder)
# name = r"$N_2(\mu, \Sigma)$ target"
# plot_tv_upper_bound(tv_f, name)
# plot_tau_stuff(tau_f, name)

# demo_folder = "MVN3-haar-cov"
# tau_f =  read_demo_df_file("tau lag 2025-02-17 Mon 21-38.csv", demo_folder)
# tv_f =  read_demo_df_file("TV est 2025-02-17 Mon 21-38.csv" , demo_folder)
# name = r"$N_3(\mu, \Sigma)$ target" 
# plot_tv_upper_bound(tv_f, name)
# plot_tau_stuff(tau_f, name)


"""
get a coupled chain from a MVN
"""

# x_chain, y_chain  = just_get_mvn_mcmc_chain(random_state = 1001)

# print(y_chain.shape, x_chain.shape )
# y_path  = os.path.join("keep_safe", "3D-MVN-sample", "lagged-chain.csv")
# x_path  = os.path.join("keep_safe", "3D-MVN-sample", "nonlagged-chain.csv")

# np.savetxt(fname = y_path,X =  y_chain, delimiter = ",")
# np.savetxt(fname = x_path,X =  x_chain, delimiter = ",")


"""
looking at lagged coupled chain that has been ran for 10x burnin of 857
target is highly correlated mvn3
"""

tau_data = read_df_file("high_autocorrelated_mvn-P3-Seed42-tau-data 2025-03-05 Wed 13-01.csv")

title = r"high autocor $N_3(\mu,\Sigma)$"
plot_tau_stuff(tau_data,title)

tv_ests = read_df_file("high_autocorrelated_mvn-P3-Seed42-tv-ests 2025-03-05 Wed 13-01.csv")
plot_tv_upper_bound(tv_ests, title)