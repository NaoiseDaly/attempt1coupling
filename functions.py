import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
logger = logging.getLogger(__name__)
from time import perf_counter
from scipy.stats import norm, uniform
from pandas import DataFrame

def print_basic_df_summary(df):
    """helper function that prints the mean, std of a dataframe"""
    print( df.apply(["mean", "std"]).agg(round, 0, 2) )

def coupled_MCMC1( max_t_iterations=10**3):
    """simulate a normal with a hardcoded mean and sd"""

    #start timing here
    start_time = perf_counter()

    x_chain = np.zeros(max_t_iterations)
    y_chain = np.zeros(max_t_iterations)
    x_chain[0], y_chain[0] = -5, 5

    def log_alpha(current, new):
        """log of the alpha probability of accepting a proposed move.
        Here the proposal dist is symmetric and the target is N(3,4)        
        """
        mu = 3
        sigma_squared = 4   
        r = (current - new)*( current+new- 2*mu )/(2*sigma_squared)
        return min(0, r )

    #conscicously using common random numbers
    log_unif_rvs = np.log(uniform.rvs(size = max_t_iterations))
    for t in range(1, max_t_iterations):
        #propose a move from maximal coupling
        #here X_t+1 is sampled from Q(X_t,.), which is a normal centered on X_t with sd 1
        x_t_1, y_t_1 = x_chain[t-1], y_chain[t-1]

        proposed_x, proposed_y = max_coupling_algo1(
            norm(x_t_1, 1).logpdf, norm(y_t_1, 1).logpdf,
            norm(x_t_1, 1).rvs, norm(y_t_1, 1).rvs
        )
        
        #sample a uniform and take log
        log_u = log_unif_rvs[t]
        
        # logger.info(
        #     f"{log_u=:.2} {proposed_y=:.1f} {log_alpha(y_t_1, proposed_y)=:.2f}"
        #     +"\n\t\t"+
        #     f"{log_u=:.2} {proposed_x=:.1f} {log_alpha(x_t_1, proposed_x)=:.2f}"
        # )
        #decide if each chain accepts or rejects the move 
        if log_u <= log_alpha(x_t_1, proposed_x):
            x_chain[t] = proposed_x
        else:
            x_chain[t] = x_t_1
        
        if log_u <= log_alpha(y_t_1, proposed_y):
            y_chain[t] = proposed_y
        else:
            y_chain[t] = y_t_1

    #end timing now
    end_time = perf_counter()
    #record timing
    logger.info(
        f"chain took {round(end_time-start_time,3)} secs to simulate {max_t_iterations} iterations"
    )

    return DataFrame({"X":x_chain, "Y":y_chain})


def max_coupling_algo1(log_p_pdf, log_q_pdf, p_sampler, q_sampler):
    """
    Sampling from a maximal coupling of x ~ p and y ~ q
    , using Chp3 Algorithm 1 from P.Jacob 2021
    
    """
    new_X = p_sampler()
    u  = uniform.rvs()
    if np.log(u) + log_p_pdf(new_X) <= log_q_pdf(new_X):
        # logger.info(f"meeting {new_X:.2f}")
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