from generate_tau_samples import *
from functions import *
import matplotlib.pyplot as plt
from matplotlib import animation

def get_univariate_coupled_chain():

    seed = 2002
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
    ax.set_xlabel("time + lag")
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
    ax.axhline()#zero line for reference
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

def animate_chains_meeting(x_chain, y_chain_shifted, title=""):

    #got to be univariate before squeezing
    assert x_chain.shape[1] == 1 and y_chain_shifted.shape[1] == 1
    x_chain, y_chain_shifted = np.squeeze(x_chain), np.squeeze(y_chain_shifted)

    fig, ax = plt.subplots()

    ax.set_ylabel("State space" )
    ax.set_xlabel("time + lag")
    if title:
        ax.set_title(title)
    else:
        ax.set_title(
            f"coupled chains with a lag of {lag}"
        )

    #need to save the lines added
    scat1 = ax.plot(x_chain[0], "r")[0]
    scat2 = ax.plot(y_chain_shifted[0], "b")[0]

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


    ani = animation.FuncAnimation(fig=fig, func=update, frames=x_chain.shape[0]-1, interval=60)
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


if __name__ == "__main__":
    x, y, lag = get_univariate_coupled_chain()
    print(x.shape,y.shape)
    animate_chains_meeting(x,y)
    animate_L2_dist_of_chains(x,y, lag)
