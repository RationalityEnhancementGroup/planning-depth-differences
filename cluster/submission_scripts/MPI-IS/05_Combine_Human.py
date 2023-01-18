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
from costometer.utils import (
    add_cost_priors_to_temp_priors,
    get_param_string,
    recalculate_maps_from_mles,
)

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
        "-s",
        "--simulated-cost-function",
        dest="simulated_cost_function",
        help="Simulated cost function YAML file",
        default="back_dist_depth_eff_forw",
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
    parser.add_argument(
        "-p",
        "--by_pid",
        dest="by_pid",
        help="If included, by pid.",
        default=True,
        action="store_false",
    )
    parser.add_argument(
        "-a",
        "--alpha",
        dest="alpha",
        help="alpha",
        type=float,
        default=1,
    )
    inputs = parser.parse_args()

    # hard coded for my folder structure
    irl_folder = Path(__file__).resolve().parents[3]
    cluster_folder = Path(__file__).resolve().parents[2]

    yaml_path = irl_folder.joinpath(
        f"data/inputs/yamls/cost_functions/{inputs.simulated_cost_function}.yaml"
    )
    with open(yaml_path, "r") as stream:
        cost_details = yaml.safe_load(stream)

    if inputs.alpha == 1:
        alpha_string = ""
    else:
        alpha_string = f"_{inputs.alpha}"

    temp_prior_details = {}
    for prior in inputs.temperature_file.split(","):
        yaml_path = irl_folder.joinpath(f"data/inputs/yamls/temperatures/{prior}.yaml")
        with open(yaml_path, "r") as stream:
            prior_inputs = yaml.safe_load(stream)
        temp_prior_details[prior] = prior_inputs

    with open(
        irl_folder.joinpath(f"cluster/parameters/cost/{inputs.cost_function}.txt"),
        "r",
    ) as f:
        full_parameters = f.read().splitlines()

    # load random file
    random_df = pd.read_csv(
        f"data/logliks/{inputs.simulated_cost_function}/"
        f"{inputs.experiment}{alpha_string}/RandomPolicy_optimization_results.csv",
        index_col=0,
    )
    random_df["applied_policy"] = "RandomPolicy"

    # load softmax files
    softmax_dfs = []
    for curr_parameters in full_parameters:
        try:
            cost_parameters = {
                cost_parameter_arg: float(arg)
                for arg, cost_parameter_arg in zip(
                    curr_parameters.split(","), cost_details["cost_parameter_args"]
                )
            }
        except ValueError as e:
            raise e

        curr_file_name = (
            f"data/logliks/{inputs.simulated_cost_function}/"
            f"{inputs.experiment}{alpha_string}/"
            f"SoftmaxPolicy_optimization_results_"
            f"{get_param_string(cost_parameters)}.csv"
        )

        curr_df = pd.read_csv(curr_file_name, index_col=0)

        curr_df["applied_policy"] = "SoftmaxPolicy"
        softmax_dfs.append(curr_df)

    softmax_dfs = pd.concat(softmax_dfs)

    full_priors = add_cost_priors_to_temp_priors(
        softmax_dfs, cost_details, temp_prior_details
    )
    softmax_dfs = recalculate_maps_from_mles(softmax_dfs, full_priors)

    full_df = pd.concat([softmax_dfs, random_df])

    if not inputs.by_pid:
        cluster_folder.joinpath(
            f"data/logliks/{inputs.cost_function}/"
            f"{inputs.experiment}{alpha_string}_by_pid/"
        ).mkdir(exist_ok=True, parents=True)

        for pid in full_df["trace_pid"].unique():
            full_df[full_df["trace_pid"] == pid].reset_index(drop=True).to_feather(
                cluster_folder.joinpath(
                    f"data/logliks/{inputs.cost_function}/"
                    f"{inputs.experiment}{alpha_string}_by_pid/{pid}.feather"
                )
            )
    else:
        cluster_folder.joinpath(f"data/logliks/{inputs.cost_function}/").mkdir(
            exist_ok=True, parents=True
        )

        full_df.reset_index(drop=True).to_feather(
            cluster_folder.joinpath(
                f"data/logliks/{inputs.cost_function}/"
                f"{inputs.experiment}{alpha_string}.feather"
            )
        )

    cluster_folder.joinpath(f"data/priors/{inputs.cost_function}").mkdir(
        parents=True, exist_ok=True
    )
    pickle.dump(
        full_priors,
        open(
            cluster_folder.joinpath(
                f"data/priors/{inputs.cost_function}/"
                f"{inputs.experiment}{alpha_string}.pkl"
            ),
            "wb",
        ),
    )
