"""
This script helps simulate trajectories using MCRL agents \
for the cluster (surprising, right?)
See more documentation in the main block.
"""
import json
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path

import dill as pickle
import numpy as np
import pandas as pd
import yaml
from cluster_utils import get_args_from_yamls
from costometer.agents import SymmetricMCLParticipant
from costometer.utils import get_param_string
from mcl_toolbox.utils.feature_normalization import get_new_feature_normalization
from mouselab.cost_functions import *  # noqa : F401

if __name__ == "__main__":
    """# noqa
    Example call:
    python simulated_mcl_trajectories.py -e high_increasing -m 1729 -f habitual -p search_space -o baseline_null -c linear_depth -v 2.0,0.0 -n 1 -t 30
    """
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
    parser.add_argument(
        "-v",
        "--values",
        dest="cost_parameter_values",
        help="Cost parameter values as comma separated string, e.g. '1.00,2.00'",
        type=str,
    )
    parser.add_argument(
        "-n",
        "--num-simulated",
        dest="num_simulated",
        help="Num simulations",
        type=int,
        default=200,
    )
    parser.add_argument(
        "-t", "--num-trials", dest="num_trials", help="Num trials", type=int, default=30
    )

    inputs = parser.parse_args()

    args = get_args_from_yamls(
        vars(inputs), attributes=["cost_function", "experiment_setting"]
    )

    path = Path(__file__).resolve().parents[2]

    # load yaml inputs
    model_attributes_file = path.joinpath(
        f"cluster/parameters/mcl/run/{inputs.model_yaml}.yaml"
    )
    with open(str(model_attributes_file), "r") as stream:
        model_attributes = yaml.safe_load(stream)

    feature_file = path.joinpath(
        f"cluster/parameters/mcl/features/{inputs.feature_yaml}.yaml"
    )
    with open(str(feature_file), "r") as stream:
        features = yaml.safe_load(stream)

    if inputs.constant_yaml:
        constant_file = path.joinpath(
            f"cluster/parameters/mcl/constant/{inputs.constant_yaml}.yaml"
        )
        with open(str(constant_file), "r") as stream:
            held_constant = yaml.safe_load(stream)
    else:
        held_constant = None

    participant_kwargs = {
        "model_attributes": model_attributes,
        "features": features["features"],
    }

    # load normalized features
    normalized_file = path.joinpath(
        f"cluster/parameters/mcl/normalized/{inputs.feature_yaml}.pkl"
    )
    if not normalized_file.is_file():
        new_normalized = get_new_feature_normalization(
            participant_kwargs["features"],
            exp_setting=inputs.experiment_setting,
            num_trials=10,
            num_simulations=10,
        )
        with open(
            str(normalized_file),
            "wb",
        ) as f:
            pickle.dump(new_normalized, f)
        participant_kwargs["normalized_features"] = new_normalized
    else:
        with open(
            str(normalized_file),
            "rb",
        ) as f:
            normalized_features = pickle.load(f)
        participant_kwargs["normalized_features"] = normalized_features

    try:
        cost_parameters = {
            cost_parameter_arg: float(arg)
            for arg, cost_parameter_arg in zip(
                inputs.cost_parameter_values.split(","), args["cost_parameter_args"]
            )
        }
    except ValueError as e:
        raise e

    with open(
        path.joinpath(
            f"data/inputs/exp_inputs/rewards/{args['ground_truth_file']}.json"
        ),
        "rb",
    ) as file_handler:
        ground_truths = json.load(file_handler)

    traces = []

    for simulation in range(inputs.num_simulated):
        ground_truth_subsets = np.random.choice(
            ground_truths, inputs.num_trials, replace=False
        )

        cost_function = eval(args["cost_function"])

        simulated_participant = SymmetricMCLParticipant(
            **deepcopy(participant_kwargs),
            num_trials=inputs.num_trials,
            cost_function=cost_function,
            cost_kwargs=cost_parameters,
            ground_truths=[trial["stateRewards"] for trial in ground_truth_subsets],
            trial_ids=[trial["trial_id"] for trial in ground_truth_subsets],
            params=held_constant,
        )
        simulated_participant.simulate_trajectory()

        trace_df = pd.DataFrame.from_dict(simulated_participant.trace)
        trace_df = trace_df.explode("actions")

        # add all information that might be useful
        for sim_param, sim_value in vars(inputs).items():
            trace_df[f"sim_{sim_param}"] = sim_value

        for cost_param, cost_val in cost_parameters.items():
            trace_df[cost_param] = cost_val

        trace_df["pid"] = simulation

        traces.append(trace_df)

    full_traces_df = pd.concat(traces)

    parameter_string = get_param_string(cost_parameters)

    path.joinpath(
        f"cluster/data/trajectories/{inputs.experiment_setting}/MCL/"
        f"{inputs.cost_function}/{inputs.prior_json}/"
    ).mkdir(parents=True, exist_ok=True)

    full_traces_df.to_csv(
        path.joinpath(
            f"cluster/data/trajectories/{inputs.experiment_setting}/MCL/"
            f"{inputs.cost_function}/{inputs.prior_json}/"
            f"{inputs.model_yaml}_{inputs.feature_yaml}_{inputs.constant_yaml}"
            f"_{parameter_string}.csv"
        )
    )
