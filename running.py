from functions import *
import logging 
import pandas as pd
logger = logging.getLogger(__name__)
logging.basicConfig( level=logging.INFO)
from scipy.stats import norm
from generate_tau_samples import modified_coupled_MCMC2, sample_tau_L_for_many_lags
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
N = 2000
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
N, L = 600, 200
# c_sample = coupled_MCMC2(lag = L,max_t_iterations= N)

# trace_plot(c_sample, L, 453)
# print_basic_df_summary(c_sample)
# print_basic_df_summary(c_sample.iloc[((N-L)//5)+L:])


results_df = sample_tau_L_for_many_lags([100*i for i in range(3,9)])
print_basic_df_summary(results_df)
plt.boxplot(results_df, tick_labels = results_df.columns)
plt.show()