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

#plot TV upper bound

tv_f = "TV est 2025-01-28 Tue 15-06.csv"#"tv est 2025-01-28 Tue 10-20.csv" 
tv_est = read_df_file(tv_f)

fig , ax1 = plt.subplots(1)
ax1.plot(tv_est)
ax1.set_title(f"TV upper bound")
ax1.legend(tv_est.columns, title = "Lag")
ax1.set_ylabel("TV upper bound")
ax1.set_xlabel("time t")
plt.show()

# d = pd.read_csv(os.path.join("logs_and_data","tau lag 2025-01-27 Mon 20-19.csv") )
tau_f = "tau lag 2025-01-28 Tue 15-06.csv" #"tau lag 2025-01-28 Tue 10-20.csv" # 
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

shifted = d.apply(lambda s:s -int(s.name), axis = 0)
print_basic_df_summary(shifted)
fig , (ax1, ax2) = plt.subplots(1,2)
ax1.boxplot(d, tick_labels = d.columns)
ax1.set_title("tau")
ax2.boxplot(shifted, tick_labels = d.columns)
ax2.set_title("tau-lag")
plt.show(	)