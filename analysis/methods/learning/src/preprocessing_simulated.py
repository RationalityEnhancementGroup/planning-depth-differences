import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import yaml
from costometer.utils import (
    add_click_count_columns_to_simulated,
    add_processed_columns,
    fix_trial_id_for_simulated,
    load_q_file,
)
from mouselab.cost_functions import *  # noqa: F401,F403

sys.path.append(str(Path(__file__).resolve().parents[4].joinpath("cluster/src")))
from cluster_utils import create_test_env  # noqa : E402

if __name__ == "__main__":
    # get arguments
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--experiment-setting",
        dest="experiment_setting",
        help="Experiment setting YAML file",
        type=str,
    )
    parser.add_argument(
        "-m",
        "--model-yaml",
        dest="model_yaml",
        help="Model YAML file",
        type=str,
    )
    parser.add_argument(
        "-f",
        "--feature-yaml",
        dest="feature_yaml",
        help="Feature YAML file",
        type=str,
    )
    parser.add_argument(
        "-p",
        "--prior-json",
        dest="prior_json",
        help="File with priors for MCL features",
        default="search_space",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--constant-yaml",
        dest="constant_yaml",
        help="Constant YAML",
        type=str,
        default="cost",
    )
    parser.add_argument(
        "-c",
        "--cost-function",
        dest="cost_function",
        help="Cost function YAML file",
        type=str,
        default="linear_depth",
    )
    inputs = parser.parse_args()

    irl_path = Path(__file__).resolve().parents[4]
    file_path = Path(__file__).resolve().parents[1]

    simulated_path = irl_path.joinpath(
        f"cluster/data/trajectories/{inputs.experiment_setting}/MCL/"
        f"{inputs.cost_function}/{inputs.prior_json}/"
    )
    simulated_file_pattern = (
        f"{inputs.model_yaml}_{inputs.feature_yaml}_{inputs.constant_yaml}"
    )

    if inputs.experiment_setting in [
        "small_test_case",
        "reduced_leaf",
        "reduced_middle",
        "reduced_root",
        "reduced_variance",
        "cogsci_learning",
        "high_increasing",
        "mini_variance",
        "zero_variance",
        "large_variance"
    ]:
        create_test_env(inputs.experiment_setting)

    file_path.joinpath(
        f"data/processed/simulated/{inputs.experiment_setting}/MCL/"
        f"{inputs.cost_function}/{inputs.prior_json}"
    ).mkdir(parents=True, exist_ok=True)

    experiment_setting_path = irl_path.joinpath(
        f"data/inputs/yamls/experiment_settings/{inputs.experiment_setting}.yaml"
    )
    with open(str(experiment_setting_path), "r") as stream:
        args = yaml.safe_load(stream)

    cost_function_path = irl_path.joinpath(
        f"data/inputs/yamls/cost_functions/{inputs.cost_function}.yaml"
    )
    with open(str(cost_function_path), "r") as stream:
        args = {**args, **yaml.safe_load(stream)}

    ground_truth_file = irl_path.joinpath(
        f"data/inputs/exp_inputs/rewards/{args['ground_truth_file']}.json"
    )
    ground_truths = json.load(open(ground_truth_file, "rb"))

    ground_truths_dict = {
        json_dict["trial_id"]: json_dict["stateRewards"] for json_dict in ground_truths
    }

    simulated_datas = []

    for curr_file in simulated_path.glob(f"{simulated_file_pattern}*"):
        curr_df = pd.read_csv(curr_file)

        cost_parameters = {
            cost_parameter_arg: float(arg)
            for arg, cost_parameter_arg in zip(
                eval(curr_df["sim_cost_parameter_values"].values[0]),
                args["cost_parameter_args"],
            )
        }

        try:
            q_dictionary = load_q_file(
                inputs.experiment_setting,
                cost_function_name=inputs.cost_function,
                cost_params=cost_parameters,
                path=irl_path.joinpath("cluster/data/q_files"),
            )
        except IndexError as e:
            print(f"No q dictionary found: {e}")
            q_dictionary = None

        curr_df["ground_truth"] = curr_df["ground_truth"].apply(eval)
        curr_df["taken_paths"] = curr_df["taken_paths"].apply(eval)
        curr_df["full_actions"] = curr_df["full_actions"].apply(eval)

        curr_df = fix_trial_id_for_simulated(curr_df, ground_truths)
        curr_df = add_click_count_columns_to_simulated(
            curr_df, args["node_classification"]
        )
        curr_df = add_processed_columns(
            curr_df,
            inputs.experiment_setting,
            ground_truths_dict,
            args["node_classification"],
            eval(inputs.cost_function),
            cost_parameters,
            human=False,
            q_dictionary=q_dictionary,
        )

        simulated_datas.append(curr_df)

    simulated_data = pd.concat(simulated_datas, ignore_index=True)
    simulated_data.to_csv(
        file_path.joinpath(
            f"data/processed/simulated/{inputs.experiment_setting}/MCL/"
            f"{inputs.cost_function}/{inputs.prior_json}/{simulated_file_pattern}.csv"
        )
    )
