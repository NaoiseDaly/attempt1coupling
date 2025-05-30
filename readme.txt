10/04/2025
Putting a capstone on things here.
This code originates from an Final Year Project looking at convergence in Markov Chains.

-   For an outside viewer, just run the open_day.py and enjoy the graphs. 
First, you will need to run `python pip install -r requirements.txt` and create a `MCMCcouplingSimulation.log` file (just rename a .txt) in a `logs_and_data` folder.

-   The jist of the FYP is the following (the terms and concepts will not be explained further):
The motivating example was using Markov Chain Monte Carlo (MCMC) on a posterior for a hierarchical model ( fitted to the 8-Schools dataset).
The distance to stationarity (of a chain's marginal distribution) can be upper bounded by a function of the meeting times of chains coupled with a lag.
This function can be estimated via Monte Carlo by running many lagged chain couplings and observing the meeting times.

-   Anything in running.py commented out may no longer work.

-   generate_tau_samples.py has both the functions for running the MCMC simulations and creating the various transition kernels.

-   functions.py holds all convience functions like plotting or I/O patterns.

-   If there is some interest in the unit tests: run create_unit_test_data.py first (a folder may need to be created) then unit_tests.py.
The tests themselves may take a couple of minutes, depending on hardware.
The principle in these tests is: If the function with the same arguments and the same seed is called multiple times, it should give the same output each time.
This says nothing about the same seed and different arguments.
One of the unit tests should in theory fail,
(Without passing a Random Number Generator - RNG object, scipy's default is to use a common system RNG. This RNG's state may go out of sync if multiple functions use it in parallel. This was also a bottleneck in performance as multiple processes were accessing the same object.)
, but empirically this has not happened.

-   remote_running.py was used to stage long running experiments on the FYP computer.

-   ignore ESS_issue.R although Vaats' paper on multivariate ESS is worthwhile for anyone interested in diagnosing convergence in MCMC.

-   the specs folder holds some material from the FYP supervisor.

-   the good_samples folder stores some experiment data with some (patchy) notes for reproducability.