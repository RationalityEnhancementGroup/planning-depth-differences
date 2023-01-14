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
        default="dist_depth_eff_forw",
    )
    parser.add_argument(
        "-o",
        "--simulated-cost-function",
        dest="simulated_cost_function",
        help="Simulated cost function YAML file",
        type=str,
        default="dist_depth_eff_forw",
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
        "--subset-value",
        dest="subset_value",
        type=float,
        default=None,
    )
    parser.add_argument(
        "-b",
        "--by_val",
        dest="by_val",
        help="If included, by sim parameter values.",
        default=True,
        action="store_false",
    )
    inputs = parser.parse_args()

    irl_folder = Path(__file__).resolve().parents[3]

    yaml_path = irl_folder.joinpath(
        f"data/inputs/yamls/cost_functions/{inputs.cost_function}.yaml"
    )

    with open(yaml_path, "r") as stream:
        cost_details = yaml.safe_load(stream)

    sim_yaml_path = irl_folder.joinpath(
        f"data/inputs/yamls/cost_functions/{inputs.simulated_cost_function}.yaml"
    )

    with open(sim_yaml_path, "r") as stream:
        sim_cost_details = yaml.safe_load(stream)

    temp_prior_details = {}
    for prior in inputs.temperature_file.split(","):
        yaml_path = irl_folder.joinpath(f"data/inputs/yamls/temperatures/{prior}.yaml")
        with open(yaml_path, "r") as stream:
            prior_inputs = yaml.safe_load(stream)
        temp_prior_details[prior] = prior_inputs

    if inputs.policy == "RandomPolicy":
        inputs.simulated_cost_function = ""

    irl_folder.joinpath(
        f"cluster/data/logliks/{inputs.cost_function}/simulated/"
        f"{inputs.experiment_setting}/"
        f"{inputs.policy}_by_pid/"
    ).mkdir(exist_ok=True, parents=True)

    all_dfs = []
    for applied_policy in ["RandomPolicy", "SoftmaxPolicy"]:
        file_pattern = (
            f"cluster/data/logliks/{inputs.cost_function}/simulated/"
            f"{inputs.experiment_setting}/{inputs.policy}/"
            f"{applied_policy}_optimization_results*"
            f"_{inputs.simulated_cost_function}*"
        )

        curr_df = pd.concat([pd.read_csv(f) for f in irl_folder.glob(file_pattern)])

        if inputs.subset_value is not None and inputs.policy == "SoftmaxPolicy":
            curr_df = curr_df[curr_df["sim_temp"] == inputs.subset_value].reset_index()
        elif inputs.subset_value:
            curr_df = curr_df[curr_df["trace_pid"] == inputs.subset_value].reset_index()

        if applied_policy == "SoftmaxPolicy":
            full_priors = add_cost_priors_to_temp_priors(
                curr_df, cost_details, temp_prior_details
            )
            curr_df = recalculate_maps_from_mles(curr_df, full_priors)

        curr_df["applied_policy"] = applied_policy
        all_dfs.append(curr_df)

    # we save files by subsetted value, if provided
    if inputs.subset_value is not None:
        subset_value_string = f"_{inputs.subset_value:.2f}"
    else:
        subset_value_string = ""
    full_df = pd.concat(all_dfs)

    if not inputs.by_val:
        for sim_cost_parameter_values in full_df["sim_cost_parameter_values"].unique():
            param_string = get_param_string(
                {
                    cost_parameter_arg: arg
                    for arg, cost_parameter_arg in zip(
                        sim_cost_parameter_values.split(","),
                        sim_cost_details["cost_parameter_args"],
                    )
                }
            )
            full_df[
                full_df["sim_cost_parameter_values"] == sim_cost_parameter_values
            ].reset_index(drop=True).to_feather(
                irl_folder.joinpath(
                    f"cluster/data/logliks/{inputs.cost_function}/simulated/"
                    f"{inputs.experiment_setting}/"
                    f"{inputs.policy}_by_pid/"
                    f"{inputs.policy}"
                    f"{'_' if len(inputs.simulated_cost_function) > 0 else ''}"
                    f"{inputs.simulated_cost_function}_applied_"
                    f"{param_string}{subset_value_string}.feather"
                )
            )
    else:
        full_df.reset_index(drop=True).to_feather(
            irl_folder.joinpath(
                f"cluster/data/logliks/{inputs.cost_function}/simulated/"
                f"{inputs.experiment_setting}/"
                f"{inputs.policy}"
                f"{'_' if len(inputs.simulated_cost_function) > 0 else ''}"
                f"{inputs.simulated_cost_function}_applied{subset_value_string}.feather"
            )
        )

    irl_folder.joinpath(f"cluster/data/priors/{inputs.cost_function}").mkdir(
        parents=True, exist_ok=True
    )
    pickle.dump(
        full_priors,
        open(
            irl_folder.joinpath(
                f"cluster/data/priors/"
                f"{inputs.cost_function}/{inputs.policy}"
                f"{'_' if len(inputs.simulated_cost_function)>0 else ''}"
                f"{inputs.simulated_cost_function}_applied.pkl"
            ),
            "wb",
        ),
    )
