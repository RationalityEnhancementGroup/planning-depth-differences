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
from cluster_utils import create_test_env, get_args_from_yamls,adjust_ground_truth,adjust_state
from costometer.utils import get_param_string
from mouselab.cost_functions import *  # noqa: F401, F403
from mouselab.envs.registry import registry
from mouselab.graph_utils import get_structure_properties
from mouselab.metacontroller.mouselab_env import MetaControllerMouselab
from mouselab.metacontroller.vanilla_BMPS import load_feature_file
from mouselab.mouselab import MouselabEnv

def get_state_action_values(
    experiment_setting: str,
    bmps_file: str,
    cost_parameters: Dict[str, float],
    cost_function: Callable,
    cost_function_name: str = None,
    structure: Dict[Any, Any] = None,
    path: Union[str, bytes, os.PathLike] = None,
    env_params: Dict[Any, Any] = None,
    alpha: float = 1,
    gamma: float = 1,
) -> Callable:
    """
    Gets BMPS weights for different cost functions

    :param experiment_setting: which experiment setting (e.g. high increasing)
    :param cost_parameters: a dictionary of inputs for the cost function
    :param cost_function: cost function to use
    :param cost_function_name:
    :param structure: where nodes are
    :param path:
    :param env_params
    :param alpha
    :param gamma
    :return: info dictionary which contains q_dictionary, \
    function additionally saves this dictionary into data/q_files
    """
    W = np.asarray([1, 1, 1])

    if env_params is None:
        env_params = {}

    env = MouselabEnv.new_symmetric_registered(
        experiment_setting,
        cost=cost_function(**cost_parameters),
        mdp_graph_properties=structure,
        **env_params,
    )

    (
        _,
        features,
        _,
        _,
    ) = load_feature_file(
        bmps_file, path=Path(__file__).parents[1].joinpath("parameters/bmps/")
    )

    env = MetaControllerMouselab(
        env.tree,
        env.init,
        term_belief=False,
        features=features,
        seed=91,
        cost=cost_function(**cost_parameters),
        mdp_graph_properties=structure,
        **env_params,
    )

    env.ground_truth = adjust_ground_truth(env.ground_truth, alpha, gamma, env.mdp_graph.nodes.data("depth"))
    env._state = adjust_state(env._state, alpha, gamma, env.mdp_graph.nodes.data("depth"))

    Q = lambda state, action: np.dot(
        env.action_features(state=state, action=action), W
    )  # noqa : E731

    info = {"q_dictionary": Q}
    # saves res dict
    if path is not None:
        parameter_string = get_param_string(cost_parameters)
        if alpha == 1:
            alpha_string = ""
        else:
            alpha_string = f"_{alpha:.2f}"

        if gamma == 1:
            gamma_string = ""
        else:
            gamma_string = f"{gamma:.3f}"

        path.joinpath(
            f"preferences/{experiment_setting}{gamma_string}{alpha_string}/{cost_function_name}/"
        ).mkdir(parents=True, exist_ok=True)
        filename = path.joinpath(
            f"preferences/{experiment_setting}{gamma_string}{alpha_string}/{cost_function_name}/"
            f"BMPS_{experiment_setting}{gamma_string}{alpha_string}_{parameter_string}.dat"  # noqa: E501
        )

        pickled_data = pickle.dumps(info)
        compressed_pickle = blosc.compress(pickled_data)

        with open(filename, "wb") as f:
            f.write(compressed_pickle)

    return Q


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
        "-a",
        "--gamma",
        dest="gamma",
        help="gamma",
        type=float,
        default=1,
    )
    parser.add_argument(
        "-g",
        "--alpha",
        dest="alpha",
        help="alpha",
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
            alpha=inputs.alpha,
            gamma=inputs.gamma,
            path=Path(__file__).parents[1].joinpath("data/bmps"),
        )
