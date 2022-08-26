# Learning Methods

This directory contains the work done for the learning extension work.

## Setting up virtual environment in  jupyter notebook

To use your virtual environment in jupyter notebooks, switch to the virtual environment and install a named kernel:
```
source env/bin/activate
python -m ipykernel install --user --name=<env>
```

To remove (if wanted):
```
jupyter kernelspec uninstall <env>
```

## Analysis

This directory contains analysis notebooks to investigate extending the IRL method to the learning period:

- `investigate_simulated_behavior.ipynb`: in this notebook we analyze various simulated MCRL agents and look at their clicking behavior over time, as well as looking at the best cost model fit for these agents
- ...

## Preliminary cluster work

To simulate learning agents using MCRL code:

```
cd <path to irl-project>/cluster
condor_submit_bid 2 submission_scripts/MPI-IS/L_03_Simulate_MCL.sub <parameters you want to run>
```

If you are not simulating too many agents, you can do that locally on your own computer, e.g.:

```
cd <path to irl-project>/cluster
python src/simulate_mcl_trajectories.py -e high_increasing -m 1729 -f habitual -p search_space -o baseline_null -c linear_depth -v 5.0,0.0 -n 3 -t 30
python src/simulate_mcl_trajectories.py -e high_increasing -m 1729 -f habitual -p search_space -o baseline_null -c linear_depth -v 0.0,5.0 -n 3 -t 30
```



