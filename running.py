from functions import *
import logging 
logger = logging.getLogger(__name__)
logging.basicConfig( level=logging.INFO)
np.random.seed(1345301)

"""looking at the maximal coupling algorithm on its own"""
# sample = [
#     max_coupling_algo1(norm(5,3).logpdf, norm(10,1).logpdf, norm(5,3).rvs, norm(10,1).rvs)
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

tv_f = "tv est 2025-02-04 Tue 13-36.csv"
#"tv est 2025-02-03 Mon 21-33.csv"#"TV est 2025-01-28 Tue 15-06.csv"#"tv est 2025-01-28 Tue 10-20.csv" 
tv_est = read_df_file(tv_f)

fig , ax1 = plt.subplots(1)
ax1.plot(tv_est)
ax1.axhline(0,color = "black", ls ="--")
ax1.set_title(f"TV upper bound")
ax1.legend(tv_est.columns, title = "Lag")
ax1.set_ylabel("TV upper bound")
ax1.set_xlabel("time t")
plt.show()

# d = pd.read_csv(os.path.join("logs_and_data","tau lag 2025-01-27 Mon 20-19.csv") )
tau_f = "tau lag 2025-02-04 Tue 13-36.csv"
#"tau lag 2025-02-03 Mon 21-33.csv"#"tau lag 2025-01-28 Tue 15-06.csv" #"tau lag 2025-01-28 Tue 10-20.csv" # 
d = read_df_file(tau_f)

fig , (ax1) = plt.subplots(1)
for c in d.columns:
    xs = d[c] -int(c) #tau - lag
    ax1.ecdf(xs, complementary  = True )

ax1.set_title("ECCDF of tau")
ax1.set_xlabel("tau - lag")
ax1.legend(d.columns, title = "lag")
ax1.set_yscale("log")
plt.show()

#show them indivivdually
for col in d.columns:
    fig , ax1 = plt.subplots(1)
    xs = d[col] -int(col) #tau - lag
    ax1.ecdf(xs, complementary  = True )

    ax1.set_title(f"ECCDF of tau when lag is {col}")
    ax1.set_xlabel("tau - lag")
    ax1.set_yscale("log")
    plt.show()

#examine tau
shifted = d.apply(lambda s:s -int(s.name), axis = 0)
print_basic_df_summary(shifted)
fig , (ax1, ax2) = plt.subplots(1,2)
ax1.boxplot(d, tick_labels = d.columns)
ax1.set_title("tau")
ax2.boxplot(shifted, tick_labels = d.columns)
ax2.set_title("tau-lag")
plt.show(	)