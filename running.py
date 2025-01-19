from functions import *
import logging 
import pandas as pd
logger = logging.getLogger(__name__)
logging.basicConfig( level=logging.INFO)
from scipy.stats import norm
np.random.seed(1345301)

"""looking at the maximal coupling algorithm on its own"""
sample = [
    max_coupling_algo1(norm(10,1).logpdf, norm(0,1).logpdf, norm(10,1).rvs, norm(0,1).rvs)
    for _ in range(200)]
df = pd.DataFrame({"X":[pair[0] for pair in sample],
                   "Y":[ pair[1] for pair in sample]})


just_plot_it(df["X"], df["Y"])
plot_joint_marginal(df)

"""Doing MCMC with the maximal coupling algorithm as the proposal dist"""
N = 2000
sample = coupled_MCMC1(N)
sample_after_burn_in = sample.iloc[N//5:]

print_basic_df_summary(sample)
print_basic_df_summary(sample_after_burn_in)

plot_joint_marginal(sample)
plot_joint_marginal(sample_after_burn_in)
