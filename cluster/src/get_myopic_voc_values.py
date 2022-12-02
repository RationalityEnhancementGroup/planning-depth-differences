"""
Script helps calculate and save Q values for Mouselab environments
"""
import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Callable, Dict, List

import blosc
import dill as pickle
import numpy as np
from cluster_utils import create_test_env, get_args_from_yamls
from contexttimer import Timer
from costometer.utils import get_param_string
from mouselab.cost_functions import *  # noqa: F401, F403
from mouselab.env_utils import (
    add_extended_state_to_sa_pairs,
    get_all_possible_states_for_ground_truths,
    get_ground_truths_from_json,
    get_sa_pairs_from_states,
)
from mouselab.exact import hash_tree
from mouselab.graph_utils import get_structure_properties
from mouselab.metacontroller.mouselab_env import MetaControllerMouselab
from mouselab.metacontroller.vanilla_BMPS import load_feature_file
from mouselab.mouselab import MouselabEnv
from toolz import unique


def get_bmps_rollouts(
    experiment_setting: str,
    bmps_file: str,
    cost_parameters: Dict[str, float],
    cost_function: Callable,
    cost_function_name: str = None,
    structure: Dict[Any, Any] = None,
    ground_truths: List[List[float]] = None,
    env_params: Dict[Any, Any] = None,
) -> Dict[Any, Any]:
    """
    Gets BMPS weights for different cost functions

    :param experiment_setting: which experiment setting (e.g. high increasing)
    :param cost_parameters: a dictionary of inputs for the cost function
    :param cost_function: cost function to use
    :param cost_function_name:
    :param structure: where nodes are
    :param ground_truths: ground truths to save
    :return: info dictionary which contains q_dictionary, \
    function additionally saves this dictionary into data/q_files
    """
    # get path to save dictionary in
    path = Path(__file__).parents[1].joinpath("data/bmps")
    path.joinpath(f"preferences/{experiment_setting}/{cost_function_name}/").mkdir(
        parents=True, exist_ok=True
    )

    W = np.asarray([1, 1, 1])

    if env_params is None:
        env_params = {}

    env = MouselabEnv.new_symmetric_registered(
        experiment_setting,
        cost=cost_function(**cost_parameters),
        mdp_graph_properties=structure,
        **env_params,
    )

    print(env_params)
    (
        optimization_kwargs,
        features,
        additional_kwargs,
        secondary_variables,
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

    # duplicated from an (in)exact solve routine

    verbose = (True,)
    hash_key = hash_tree
    dedup_by_hash = False

    with Timer() as timer:
        states = get_all_possible_states_for_ground_truths(
            categorical_gym_env=env, ground_truths=ground_truths
        )
        sa_pairs = get_sa_pairs_from_states(states)
        if env.include_last_action:
            sa_pairs = add_extended_state_to_sa_pairs(sa_pairs)

        if verbose:
            print(f"Got (s,a) pairs, time elapsed: {timer.elapsed}")

    if verbose:
        print(f"Deduping (s,a) to save, time elapsed: {timer.elapsed}")
    if dedup_by_hash and hash_key:
        sa_pairs = unique(sa_pairs, key=lambda sa_pair: hash_key(env, *sa_pair))
    else:
        sa_pairs = unique(sa_pairs)

    Q = {}
    for state, action in sa_pairs:
        Q[(state, action)] = np.dot(env.action_features(state=state, action=action), W)

    info = {"q_dictionary": Q}
    # saves res dict
    if path is not None:
        parameter_string = get_param_string(cost_parameters)
        path.joinpath(f"preferences/{experiment_setting}/{cost_function_name}/").mkdir(
            parents=True, exist_ok=True
        )
        filename = path.joinpath(
            f"preferences/{experiment_setting}/{cost_function_name}/"
            f"BMPS_{experiment_setting}_{parameter_string}.dat"  # noqa: E501
        )

        pickled_data = pickle.dumps(info)
        compressed_pickle = blosc.compress(pickled_data)

        with open(filename, "wb") as f:
            f.write(compressed_pickle)

    print(info)
    return info


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
        "-v",
        "--values",
        dest="cost_parameter_values",
        help="Cost parameter values as comma separated string, e.g. '1.00,2.00'",
        type=str,
    )

    inputs = parser.parse_args()
    args = get_args_from_yamls(
        vars(inputs), attributes=["cost_function", "experiment_setting"]
    )

    experiment_setting = args["experiment_setting"]

    # test setting unique to this work
    if experiment_setting in [
        "small_test_case",
        "reduced_leaf",
        "reduced_middle",
        "reduced_root",
        "reduced_variance",
    ]:
        create_test_env(experiment_setting)

    if args["ground_truth_file"]:
        ground_truths = get_ground_truths_from_json(
            Path(__file__)
            .resolve()
            .parents[2]
            .joinpath(
                f"data/inputs/exp_inputs/rewards/{args['ground_truth_file']}.json"
            )
        )
    else:
        ground_truths = None

    try:
        cost_parameters = {
            cost_parameter_arg: float(arg)
            for arg, cost_parameter_arg in zip(
                inputs.cost_parameter_values.split(","), args["cost_parameter_args"]
            )
        }
    except ValueError as e:
        raise e
    print(cost_parameters)

    cost_function = eval(args["cost_function"])

    if "structure" in args:
        with open(
            Path(__file__)
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

    get_bmps_rollouts(
        experiment_setting,
        inputs.bmps_file,
        cost_parameters,
        structure=structure_dicts,
        cost_function=cost_function,
        cost_function_name=cost_function_name,
        env_params=args["env_params"],
        ground_truths=ground_truths,
    )
