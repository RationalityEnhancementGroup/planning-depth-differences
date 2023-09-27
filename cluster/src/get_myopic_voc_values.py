"""
Script helps calculate and save Q values for Mouselab environments
"""
import json
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Callable, Dict, Union

import blosc
import dill as pickle
import numpy as np
from cluster_utils import create_test_env, get_args_from_yamls
from costometer.utils import (
    adjust_ground_truth,
    adjust_state,
    get_param_string,
    get_state_action_values,
)
from mouselab.cost_functions import *  # noqa: F401, F403
from mouselab.envs.registry import registry
from mouselab.graph_utils import get_structure_properties
from mouselab.metacontroller.mouselab_env import MetaControllerMouselab
from mouselab.metacontroller.vanilla_BMPS import load_feature_file
from mouselab.mouselab import MouselabEnv

if __name__ == "__main__":
    """
    Example call: python src/get_q_values.py -s experiment_setting -c cost \
    -v cost_parameter_values
    """
    # get arguments
    parser = ArgumentParser()
    parser.add_argument(
        "-s",
        "--experiment-setting",
        dest="experiment_setting",
        help="Experiment setting YAML file",
        type=str,
    )
    parser.add_argument(
        "-b",
        "--bmps-file",
        dest="bmps_file",
        default="Basic",
        help="BMPS Features and Optimization",
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
        "-f",
        "--cost-file",
        dest="cost_parameter_file",
        help="Cost parameter file located in cluster/parameters/cost",
        type=str,
    )
    parser.add_argument(
        "-g",
        "--gamma",
        dest="gamma",
        help="gamma",
        type=float,
        default=1,
    )
    parser.add_argument(
        "-a",
        "--kappa",
        dest="kappa",
        help="kappa",
        type=float,
        default=1,
    )

    inputs = parser.parse_args()
    args = get_args_from_yamls(
        vars(inputs), attributes=["cost_function", "experiment_setting"]
    )

    experiment_setting = args["experiment_setting"]

    try:
        registry(experiment_setting)
    except:  # noqa: E722
        create_test_env(experiment_setting)

    cost_function = eval(args["cost_function"])

    if "structure" in args:
        with open(
            Path(__file__)
            .resolve()
            .parents[2]
            .joinpath(f"data/inputs/exp_inputs/structure/{args['structure']}.json"),
            "rb",
        ) as f:
            structure_data = json.load(f)

        structure_dicts = get_structure_properties(structure_data)
    else:
        structure_dicts = None

    if callable(eval(args["cost_function"])):
        cost_function_name = inputs.cost_function
    else:
        cost_function_name = None

    with open(
        Path(__file__)
        .parents[1]
        .joinpath(f"parameters/cost/{inputs.cost_parameter_file}.txt"),
        "r",
    ) as f:
        full_parameters = f.read().splitlines()

    for curr_parameters in full_parameters:
        try:
            cost_parameters = {
                cost_parameter_arg: float(arg)
                for arg, cost_parameter_arg in zip(
                    curr_parameters.split(","), args["cost_parameter_args"]
                )
            }
        except ValueError as e:
            raise e

        get_state_action_values(
            experiment_setting,
            inputs.bmps_file,
            cost_parameters,
            structure=structure_dicts,
            cost_function=cost_function,
            cost_function_name=cost_function_name,
            env_params=args["env_params"],
            kappa=inputs.kappa,
            gamma=inputs.gamma,
            path=Path(__file__).parents[1].joinpath("data/bmps"),
        )
