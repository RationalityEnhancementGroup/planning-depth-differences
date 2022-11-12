from argparse import ArgumentParser
from pathlib import Path

import dill as pickle
import numpy as np
import pandas as pd

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--experiment",
        dest="experiment",
        help="Experiment",
        type=str,
    )
    parser.add_argument(
        "-m",
        "--max-evals",
        dest="max_evals",
        help="Max number of evaluations",
        default=300,
        type=int,
    )
    parser.add_argument(
        "-n",
        "--experiment",
        dest="num_simulations",
        help="Number of simulations (set to 1 if optimization criterion is likelihood)",
        default=30,
        type=int,
    )
    parser.add_argument(
        "-c",
        "--optimization-criterion",
        dest="optimization_criterion",
        help="Optimization criterion",
        choices=[
            "likelihood",
            "pseudo_likelihood",
            "mer_performance_error",
            "performance_error",
            "clicks_overlap",
        ],
        default="likelihood",
        type=str,
    )
    inputs = parser.parse_args()
    path = Path(__file__).resolve().parents[2]

    # likelihood doesn't use simulations > 1
    if inputs.optimization_criterion == "likelihood":
        inputs.num_simulations = 1

    # TODO not use model index

    # hard coded for my folder structure
    path = Path(__file__).resolve().parents[2]
    prior_directory = path.joinpath(
        f"cluster/data/results/mcrl/{inputs.experiment}_priors"
    )
    # make directory if it doesn't already exist
    prior_directory.mkdir(parents=True, exist_ok=True)

    mouselab_data = pd.read_csv(
        path.joinpath(f"data/processed/{inputs.experiment}/mouselab-mdp.csv")
    )
    pid_range = np.unique(mouselab_data["pid"])

    all_priors = {}

    for pid in pid_range:
        pid_prior_file = prior_directory.joinpath(
            f"{pid}_{optimization_criterion}_{max_evals}_{num_simulations}_{model_index}.pkl"  # noqa
        )
        if pid_prior_file.exists():
            with open(pid_prior_file, "rb") as file_handler:
                pid_priors = pickle.load(file_handler)
            all_priors[pid] = pid_priors
        else:
            print(pid_prior_file)

    prior_file = prior_directory.joinpath(
        f"{inputs.experiment}_{optimization_criterion}_{max_evals}_{num_simulations}_{model_index}.pkl"  # noqa
    )
    with open(prior_file, "wb") as file_handler:
        pickle.dump(all_priors, file_handler)
