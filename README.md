# planning-depth-differences

This is code and helper scripts for running Bayesian inverse reinforcement learning on Mouselab-MDP data.

# How to get started?

We recommend using a virtual environment. There is a provided requirements.txt.

These instructions work on the MPI-IS cluster and should work on any Mac/OSX system with virtualenv on it. See here for how to install virtualenv/commands on Windows: https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/.

```
#create virtual environment
python3 -m venv env
#activate virtual environment
source env/bin/activate
#install requirements
python -m pip install -r requirements.txt
```

If you will preprocess the data, add the virtual environment as an ipython kernel:
```
cd <path to planning-depth-differences>
source env/bin/activate
python -m ipykernel install --user --name=planning-depth-differences
```

# How to run?

See more on how to run cluster jobs under `cluster/README.md`.

Analysis scripts should all be well documented in their respective subdirectories in `analysis/`.

The project structure:

```
planning-depth-differences/
├── analysis
│   └── methods
│       └── static <- analysis code for paper
├── cluster
│   ├── src <- cluster scripts
│   └── submission_scripts
│       └── MPI-IS <- submission files for cluster
├── data
│   ├── hit_ids <- where to put experiment HIT IDs for downloading data
│   ├── inputs
│   │   ├── exp_inputs <- inputs for experiment (e.g., web of cash structure)
│   │   └── yamls <- inputs and details for cost functions and experiments
│   ├── processed <- processed data
│   ├── raw <- raw data
│   ├── src <- functions for downloading and processing data
│   └── templates <- jupyter notebook template for displaying information for processed data
```

# Citation

#TODO
