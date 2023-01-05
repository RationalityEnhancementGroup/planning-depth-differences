"""
This script combines all the inference files (one for each considered cost weight  \
combination) into one big (but smaller than it would be as a pickle) feather file.
It also will add in the prior probability for cost parameters for MAP estimates.
"""
from argparse import ArgumentParser
from pathlib import Path

import dill as pickle
import pandas as pd
import yaml
from costometer.utils import add_cost_priors_to_temp_priors, recalculate_maps_from_mles

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
        "-c",
        "--cost-function",
        dest="cost_function",
        help="Cost function YAML file",
        type=str,
    )
    parser.add_argument(
        "-t",
        "--temperature-file",
        dest="temperature_file",
        help="File with temperatures to infer over",
        type=str,
        default="expon,uniform",
    )
    inputs = parser.parse_args()

    # hard coded for my folder structure
    irl_folder = Path(__file__).resolve().parents[3]
    cluster_folder = Path(__file__).resolve().parents[2]

    yaml_path = irl_folder.joinpath(
        f"data/inputs/yamls/cost_functions/{inputs.cost_function}.yaml"
    )

    with open(yaml_path, "r") as stream:
        cost_details = yaml.safe_load(stream)

    temp_prior_details = {}
    for prior in inputs.temperature_file.split(","):
        yaml_path = irl_folder.joinpath(f"data/inputs/yamls/temperatures/{prior}.yaml")
        with open(yaml_path, "r") as stream:
            prior_inputs = yaml.safe_load(stream)
        temp_prior_details[prior] = prior_inputs

    Path(f"data/logliks/{inputs.cost_function}/" f"{inputs.experiment}_by_pid/").mkdir(
        exist_ok=True, parents=True
    )

    all_dfs = []
    for applied_policy in ["RandomPolicy", "SoftmaxPolicy"]:
        file_pattern = (
            f"data/logliks/{inputs.cost_function}/"
            f"{inputs.experiment}/{applied_policy}*.csv"
        )

        curr_df = pd.concat(
            [
                pd.read_csv(file, index_col=0)
                for file in cluster_folder.glob(file_pattern)
            ]
        )

        if applied_policy == "SoftmaxPolicy":
            full_priors = add_cost_priors_to_temp_priors(
                curr_df, cost_details, temp_prior_details
            )
            curr_df = recalculate_maps_from_mles(curr_df, full_priors)

        curr_df["applied_policy"] = applied_policy
        all_dfs.append(curr_df)

    full_df = pd.concat(all_dfs)

    for pid in full_df["trace_pid"].unique():
        full_df[full_df["trace_pid"] == pid].reset_index(drop=True).to_feather(
            f"data/logliks/{inputs.cost_function}/"
            f"{inputs.experiment}_by_pid/{pid}.feather"
        )

    cluster_folder.joinpath(f"data/priors/{inputs.cost_function}").mkdir(
        parents=True, exist_ok=True
    )
    pickle.dump(
        full_priors,
        open(
            cluster_folder.joinpath(
                f"data/priors/{inputs.cost_function}/{inputs.experiment}.pkl"
            ),
            "wb",
        ),
    )
