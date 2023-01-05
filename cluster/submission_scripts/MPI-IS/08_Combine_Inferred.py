"""
This script combines all the inference files (one for each considered cost weight  \
combination) into one big (but smaller than it would be as a pickle) feather file.
It also will add in the prior probability for cost parameters for MAP estimates.
"""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import yaml
from costometer.utils import add_cost_priors_to_temp_priors, recalculate_maps_from_mles

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-p",
        "--policy",
        dest="policy",
        help="Policy to simulate",
        choices=["OptimalQ", "SoftmaxPolicy", "RandomPolicy"],
        type=str,
    )
    parser.add_argument(
        "-c",
        "--cost-function",
        dest="cost_function",
        help="Cost function YAML file",
        type=str,
        default="dist_depth_forw",
    )
    parser.add_argument(
        "-o",
        "--simulated-cost-function",
        dest="simulated_cost_function",
        help="Simulated cost function YAML file",
        type=str,
        default="dist_depth_forw",
    )
    parser.add_argument(
        "-e",
        "--experiment-setting",
        dest="experiment_setting",
        help="Experiment setting YAML file",
        type=str,
        default="high_increasing",
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
        "-v",
        "--simulated-temperature",
        dest="simulated_temperature",
        type=float,
        default=None,
    )
    inputs = parser.parse_args()

    irl_folder = Path(__file__).resolve().parents[3]

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

    if inputs.policy == "RandomPolicy":
        inputs.simulated_cost_function = ""

    all_dfs = []
    for applied_policy in ["RandomPolicy", "SoftmaxPolicy"]:
        file_pattern = (
            f"cluster/data/logliks/{inputs.cost_function}/simulated/"
            f"{inputs.experiment_setting}/{inputs.policy}/"
            f"{applied_policy}_optimization_results*simulated_agents"
            f"_{inputs.simulated_cost_function}*"
        )

        curr_df = pd.concat([pd.read_csv(f) for f in irl_folder.glob(file_pattern)])

        if inputs.simulated_temperature is not None:
            curr_df = curr_df[
                curr_df["sim_temp"] == inputs.simulated_temperature
            ].reset_index()

        if applied_policy == "SoftmaxPolicy":
            full_priors = add_cost_priors_to_temp_priors(
                curr_df, cost_details, temp_prior_details
            )
            curr_df = recalculate_maps_from_mles(curr_df, full_priors)

        curr_df["applied_policy"] = applied_policy
        all_dfs.append(curr_df)

    # we save files by temperature, if simulated temp provided
    if inputs.simulated_temperature is not None:
        temp_string = f"_{inputs.simulated_temperature:.2f}"
    else:
        temp_string = ""
    pd.concat(all_dfs).reset_index(drop=True).to_feather(
        irl_folder.joinpath(
            f"cluster/data/logliks/{inputs.cost_function}/simulated/"
            f"{inputs.experiment_setting}/"
            f"{inputs.policy}{'_' if len(inputs.simulated_cost_function)>0 else ''}"
            f"{inputs.simulated_cost_function}_applied{temp_string}.feather"
        )
    )
