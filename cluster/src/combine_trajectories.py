"""This script helps combine simulated trajectories, which can be useful for \
generating heatmaps of click types and comparing simulated data to human data"""
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

if __name__ == "__main__":
    """
    Combines many simulated agents.
    Example usages:

    1. python src/combine_trajectories.py -p SoftmaxPolicy -c linear_depth -t 10.00
    2. python src/combine_trajectories.py -p OptimalQ -c linear_depth -t 10.00
    """
    parser = ArgumentParser()
    parser.add_argument(
        "-s",
        "--experiment-setting",
        dest="experiment_setting",
        help="Experiment setting YAML file",
        type=str,
    )
    parser.add_argument(
        "-c",
        "--cost-function",
        dest="cost_function",
        help="Cost function YAML file",
        type=str,
    )
    parser.add_argument(
        "-p",
        "--policy",
        dest="policy",
        help="Policy function",
        type=str,
        default=None,
    )
    inputs = parser.parse_args()

    path = Path(__file__).resolve().parents[1]
    trajectory_path = path.joinpath("data/trajectories")

    if inputs.policy == "RandomPolicy":
        inputs.cost_function = ""

    experiment = (
        f"{inputs.experiment_setting}/{inputs.policy}/"
        f"simulated_agents_{inputs.cost_function}*"
    )

    sim_files = trajectory_path.glob(experiment)
    mouselab_data = pd.concat(
        [pd.read_csv(sim_file, index_col=0) for sim_file in sim_files]
    )
    if inputs.policy == "RandomPolicy":
        save_path = inputs.policy
    else:
        save_path = f"{inputs.policy}_{inputs.cost_function}"

    path.joinpath(f"data/{inputs.policy}/").mkdir(parents=True, exist_ok=True)

    mouselab_data.to_csv(path.joinpath(f"data/{inputs.policy}/{save_path}.csv"))
