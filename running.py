from functions import *
import logging 
import pandas as pd
logger = logging.getLogger(__name__)
logging.basicConfig( level=logging.INFO)
from scipy.stats import norm
np.random.seed(42421)

sample = [
    max_coupling_algo1(norm(10,1).logpdf, norm(0,1).logpdf, norm(10,1).rvs, norm(0,1).rvs)
    for _ in range(200)]
df = pd.DataFrame({"X":[pair[0] for pair in sample],
                   "Y":[ pair[1] for pair in sample]})


# just_plot_it(x_sample, y_sample)
plot_joint_marginal(df)
