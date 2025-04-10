from generate_tau_samples import *
from functions import *
import matplotlib.pyplot as plt
from matplotlib import animation

def get_univariate_coupled_chain(seed):

    mcmc_obj = Some_random_Pd_mcmc(1,seed)
    lag = 100
    print(mcmc_obj)
    x_chain, y_chain = mcmc_obj.generate_mcmc_sample(
        lag = lag
        ,max_t_iterations =500
        ,return_chain = True
        ,random_state = seed
    )

    y_chain_shifted = np.pad(y_chain,((lag,0),(0,0)), mode = "constant", constant_values = np.nan)

    return x_chain, y_chain_shifted, lag

def animate_L2_dist_of_chains(x_c, y_c, lag, title = ""):

    fig, ax = plt.subplots()
    ax.set_ylabel(
        r"$L_2$ Norm"
    )
    ax.set_xlabel("time")
    if title:
        ax.set_title(title)
    else:
        ax.set_title(
            f"Distance between coupled chains with lag {lag}"
        )


    dists = np.linalg.norm(x_c-y_c, axis = 1)
   
    times = np.arange(dists.shape[0])[~np.isnan(dists)]
    dists = dists[~np.isnan(dists)]

    #lines
    ax.axhline(color = "black", ls = "--")#zero line for reference
    dist_line = ax.plot(times[0], dists[0], "r-")[0]


    ax.set(xlim = (0,1), ylim = (0, 1))#guess

    def update(frame):
        frame+=1#dont want zero indexed

        new_dists = dists[:frame]
        dist_line.set_data(times[:frame],new_dists)

        #get a bit of space either side of the line
        ylim_max = new_dists.max()
        ylim_max = ylim_max + .1*abs(ylim_max)

        ylim_min  = new_dists.min()
        if ylim_min == 0:
            #want a bit of space underneath the zero line
            ylim_min -= .1*abs(ylim_max)
        else:
            ylim_min = ylim_min - .1 * abs(ylim_min)

        ax.set(xlim = (times[0], times[frame]+1)
                ,ylim = (ylim_min, ylim_max))
            
        return (dist_line)

    ani = animation.FuncAnimation(fig=fig, func=update, frames=dists.shape[0]-1, interval=60)
    plt.show()

def animate_univariate_chains_meeting(x_chain, y_chain_shifted, title=""):

    #got to be univariate
    assert x_chain.shape[1] == 1 and y_chain_shifted.shape[1] == 1
    x_chain, y_chain_shifted = np.squeeze(x_chain), np.squeeze(y_chain_shifted)

    fig, ax = plt.subplots()

    #need to save the lines added
    scat1 = ax.plot(x_chain[0] ,color = "red", label = "Unlagged chain")[0]
    scat2 = ax.plot(y_chain_shifted[0], color ="blue",linestyle='dashed'
                    ,label = "Lagged chain")[0]

    leg = ax.legend()

    def init():
        ax.set_ylabel("State space" )
        ax.set_xlabel("time")
        if title:
            ax.set_title(title)

        scat1.set_color("red")
        scat2.set_color("blue")
        

        y_min = x_chain[0]
        y_max = x_chain[0]
        y_min = y_min - .1*abs(y_min)
        y_max = y_max + .1*abs(y_max)
        ax.set(xlim = (0,1), ylim = (y_min, y_max))

    def update(frame):
        frame+=1 # frame is 0 indexed, cba dealing with empty arrays
        new_x_chain = x_chain[:frame]
        new_y_chain = y_chain_shifted[:frame]
        # print(new_x, new_y)

        scat1.set_data(range(frame), new_x_chain)
        scat2.set_data(range(frame), new_y_chain)
        
        if new_x_chain[-1] == new_y_chain[-1]:
            scat1.set_color("purple")
            scat2.set_color("purple")
            #change the legend as well
            for line in leg.get_lines():
                line.set_color("purple")

        #so a np.nan seems to be nilpotent which makes a min/max call useless
        #an empty array also seems to break a min/max call
        if np.isnan(y_chain_shifted[frame-1]):
            y_min = new_x_chain.min()
            y_max= new_x_chain.max()
        else:
            temp = new_y_chain[~np.isnan(new_y_chain)]
            y_min = min(temp.min(), new_x_chain.min())
            y_max = max(new_x_chain.max(), temp.max())

        #just get a bit of space either side
        y_min = y_min - .1*abs(y_min)
        y_max = y_max + .1*abs(y_max)
        x_max = frame+1
        ax.set(xlim = (0,x_max), ylim = (y_min, y_max))

        return (scat1, scat2)


    ani = animation.FuncAnimation(fig=fig, func=update, init_func = init
                                  , frames=x_chain.shape[0]-1, interval=60)
    plt.show()


def plt_example():
    fig, ax = plt.subplots()
    t = np.linspace(0, 3, 40)
    g = -9.81
    v0 = 12
    z = g * t**2 / 2 + v0 * t

    v02 = 5
    z2 = g * t**2 / 2 + v02 * t

    scat = ax.scatter(t[0], z[0], c="b", s=5, label=f'v0 = {v0} m/s')
    line2 = ax.plot(t[0], z2[0], label=f'v0 = {v02} m/s')[0]
    ax.set(xlim=[0, 3], ylim=[-4, 10], xlabel='Time [s]', ylabel='Z [m]')
    ax.legend()


    def update(frame):
        # for each frame, update the data stored on each artist.
        x = t[:frame]
        y = z[:frame]
        # update the scatter plot:
        data = np.stack([x, y]).T
        scat.set_offsets(data)
        # update the line plot:
        line2.set_xdata(t[:frame])
        line2.set_ydata(z2[:frame])
        return (scat, line2)


    ani = animation.FuncAnimation(fig=fig, func=update, frames=40, interval=60)
    plt.show()

def last_chapter():

    #read in TV est
    folder = "8schools example with rep"
    tv_ests = read_demo_df_file("better_estimates_2chains_8schools TV est 2025-03-27 Thu 16-58.csv"
                                ,folder)
    tv_dist_title = "Estimated distance to stationarity for 8 schools posterior"
    plot_tv_upper_bound_t_long_short(tv_ests, tv_dist_title)

    biggest_lag = str(max(int(col) for col in tv_ests.columns))
    tv_bound = tv_ests[biggest_lag]
    num_ts = len(tv_bound)

    #the TV is non increasing so this is safe
    t_short = tv_bound[tv_bound <=.25].first_valid_index()
    if t_short is None:
        t_short = num_ts #not correct but surely wont happen
    t_long = tv_bound[tv_bound <=(1-.99)].first_valid_index()
    if t_long is None:
        t_long = num_ts #so things dont break if num_ts was too small
    #519 1408
    t_l, t_s = r"$t^{\text{long}}$", r"$t^{\text{short}}$"

    #read in t^long, t^short chains
    long_chain = read_good_sample_np_csv("8schools_rep_good_long_chain_2025-03-27 Thu 23-45.csv"
                                         ,"8schools example with rep")
    short_chain = read_good_sample_np_csv("8schools_rep_good_short_chain_2025-03-27 Thu 23-45.csv"
                                         ,"8schools example with rep")
    
    comp_names = [rf"$\theta_{j}$" for j in range(1,8+1) ]
    comp_names.extend([r"$\mu$", r"$\tau$"])
    trace_plot_10_comps(short_chain, comp_names)
    trace_plot_10_comps(long_chain, comp_names)
    
    #show population parameters boxplot
    short_bxp = make_boxplot_stats_from_quantiles(
        make_boxplot_quantiles(short_chain[:,8:])
    )
    long_bxp = make_boxplot_stats_from_quantiles(
        make_boxplot_quantiles(long_chain[:,8:])
    )
    boxplot_two_chains_side_by_side2(long_bxp, short_bxp, dim =2
                                     ,a_name = t_l, b_name = t_s
                                     ,var_names = comp_names[8:]
                                     ,title = "Distribution of population parameters\nsampled using different burn in points"
                                     )

    #show group parameters boxplots
    short_bxp = make_boxplot_stats_from_quantiles(
        make_boxplot_quantiles(short_chain[:,:8])
    )
    long_bxp = make_boxplot_stats_from_quantiles(
        make_boxplot_quantiles(long_chain[:,:8])
    )
    boxplot_two_chains_side_by_side2(long_bxp, short_bxp, dim =8
                                     ,a_name = t_l, b_name = t_s
                                     ,var_names = comp_names[:8]
                                     ,title = "Distribution of group parameters\nsampled using different burn in points"
                                     )
    

    
    #show marginal hist of tau
    tau_ind = 9
    fig, ax = plt.subplots()
    ax.plot(long_chain[:,tau_ind], color='green', linestyle='-')
    ax.set_title(rf"Trace plot for $\tau$")
    plt.show()
    mu_ind = 8
    fig, ax = plt.subplots()
    ax.plot(long_chain[:,mu_ind], color='green', linestyle='-')
    ax.set_title(rf"Trace plot for $\mu$")
    plt.show()

    fig, (ax1,ax2) = plt.subplots(1,2,sharex = True, sharey=True)
    num_bins = 10
    ax2.hist(short_chain[:,tau_ind],bins = num_bins, color= "bisque", density = True)
    ax1.hist(long_chain[:,tau_ind],bins = num_bins, color= "violet",density=True )

    ax2.set_title(rf"Using {t_s} burn in")
    ax2.set_xlabel(r"$\tau$")
    ax1.set_title(rf"Using {t_l} burn in")
    ax1.set_xlabel(r"$\tau$")
    
    fig.supylabel(r"Density")
    fig.suptitle(r"Marginal posterior density for $\tau$", fontsize =16)
    plt.show()

    #redoing the TV bound with more tua samples-> less Monte Carlo error -> smoother lines
    tv_ests = read_demo_df_file("better_estimates_2chains_8schools TV est 2025-04-10 Thu 12-11.csv"
                                ,folder)
    plot_tv_upper_bound_t_long_short(tv_ests, tv_dist_title, t_long = .02)#shush


if __name__ == "__main__":
    x, y, lag = get_univariate_coupled_chain(0)     # with a seed of 0 the initialisation is good
    fig, ax = plt.subplots(1)
    ax.plot(x)
    ax.set_title(r"Markov chain with a target dist of $N(\mu=27, \sigma^2=4)$")
    ax.set_ylabel("State space" )
    ax.set_xlabel("time")
    plt.show()

    title = fr"Coupled chains with lag {lag} targeting $N(\mu=27, \sigma^2=4)$"
    animate_univariate_chains_meeting(x,y, title)
    animate_L2_dist_of_chains(x,y, lag, "Distance between the coupled chains")

    y = np.genfromtxt(
        os.path.join("keep_safe","3D-MVN-sample","lagged-chain.csv")
        ,delimiter = ","
    )
    x = np.genfromtxt(
        os.path.join("keep_safe","3D-MVN-sample","nonlagged-chain.csv")
        ,delimiter = ","
    )
    lag = 500
    y = np.pad(y,((lag,0),(0,0)), mode = "constant", constant_values = np.nan)
    title = rf"Coupling with lag {lag} targeting highly autocorrelated $N_3(\mu,\Sigma)$"
    animate_L2_dist_of_chains(x,y,lag, title)


    tv_ests = read_demo_df_file(
        "TV est 2025-02-17 Mon 21-38.csv"
        ,"MVN3-haar-cov"
    )
    title = r"Estimated distance to stationarity for highly autocorrelated $N_3(\mu,\Sigma)$"
    plot_tv_upper_bound(tv_ests, title)
    
    last_chapter()
    
