import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
logger = logging.getLogger(__name__)
from time import perf_counter
from scipy.stats import norm, uniform




def max_coupling_algo1(log_p_pdf, log_q_pdf, p_sampler, q_sampler):
    """
    Sampling from a maximal coupling of x ~ p and y ~ q, using Algorithm 1 from P.Jacob 2021
    
    """
    new_X = p_sampler()
    u  = uniform.rvs()
    if np.log(u) + log_p_pdf(new_X) <= log_q_pdf(new_X):
        return (new_X, new_X) # X=Y
    
    new_Y = None
    while not new_Y:
        proposed_Y = q_sampler()
        u  = uniform.rvs()
        if np.log(u) + log_q_pdf(proposed_Y) > log_p_pdf(proposed_Y):
            new_Y = proposed_Y

    return (new_X, new_Y)


def just_plot_it(x, y,  title = None ):

    fig, ax = plt.subplots()

    ax.scatter(x,y)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if title:
        plt.suptitle(title)
    plt.show()

def plot_joint_marginal(df, x_label = "X", y_label = "Y"):
    z = sns.JointGrid(df, x = x_label, y = y_label)
    z.plot_joint(sns.scatterplot)
    z.plot_marginals(sns.histplot, kde = True, color = "r")
    plt.show()