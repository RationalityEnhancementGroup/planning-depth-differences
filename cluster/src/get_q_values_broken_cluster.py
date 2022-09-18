"""
Script helps calculate and save Q values for Mouselab environments
Made for broken cluster where we cannot submit as many jobs simultaneously
"""
import json
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Callable, Dict, List

from cluster_utils import create_test_env, get_args_from_yamls
from costometer.utils import save_q_values_for_cost
from mouselab.cost_functions import *  # noqa: F401, F403
from mouselab.env_utils import get_ground_truths_from_json
from mouselab.graph_utils import get_structure_properties


def get_q_values(
    experiment_setting: str,
    cost_parameters: Dict[str, float],
    cost_function: Callable,
    cost_function_name: str = None,
    structure: Dict[Any, Any] = None,
    ground_truths: List[List[float]] = None,
    env_params: Dict[Any, Any] = None,
) -> Dict[Any, Any]:
    """
    Gets Q values for parameter setting for linear depth cost function

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
    location = Path(__file__).parents[1].joinpath("data/q_files")
    # make directory if it doesn't already exist
    location.mkdir(parents=True, exist_ok=True)

    if env_params is None:
        env_params = {}

    info = save_q_values_for_cost(
        experiment_setting,
        cost_function=cost_function,
        cost_function_name=cost_function_name,
        cost_params=cost_parameters,
        structure=structure,
        ground_truths=ground_truths,
        path=location,
        **env_params,
    )
    return info


if __name__ == "__main__":
    """
    Example call: python src/get_q_values.py -s experiment_setting -c cost \
    -v cost_parameter_values
    """
    # get arguments
    parser = ArgumentParser()
    parser.add_argument(
        "-b",
        "--bid",
        dest="bid",
        help="bid",
        type=int,
        default=2,
    )
    parser.add_argument(
        "-s",
        "--experiment-setting",
        dest="experiment_setting",
        default="high_increasing",
        help="Experiment setting YAML file",
        type=str,
    )
    parser.add_argument(
        "-c",
        "--cost-function",
        dest="cost_function",
        default="back_dist_depth_eff_forw",
        help="Cost function YAML file",
        type=str,
    )
    parser.add_argument(
        "-n",
        "--file-number",
        dest="file_number",
        help="Cost parameter file number",
        type=int,
    )
    parser.add_argument(
        "-l",
        "--value-line",
        dest="value_line",
        help="Cost parameter values line",
        type=int,
    )
    parser.add_argument(
        "-f",
        "--file-lambda",
        dest="file_lambda",
        help="Cost parameter file lambda function",
        default='lambda num : f"params_full_five{num}.txt"',
        type=str,
    )

    inputs = parser.parse_args()
    args = get_args_from_yamls(
        vars(inputs), attributes=["cost_function", "experiment_setting"]
    )

    with open(
        Path(__file__)
        .parents[1]
        .joinpath("parameters/cost/" + eval(inputs.file_lambda)(inputs.file_number)),
        "r",
    ) as f:
        full_parameters = f.read().splitlines()
    inputs.cost_parameter_values = full_parameters[inputs.value_line]

    experiment_setting = args["experiment_setting"]

    # test setting unique to this work
    if experiment_setting == "small_test_case":
        create_test_env()

    if args["ground_truth_file"]:
        ground_truths = get_ground_truths_from_json(
            Path(__file__)
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

    get_q_values(
        experiment_setting,
        cost_parameters,
        structure=structure_dicts,
        ground_truths=ground_truths,
        cost_function=cost_function,
        cost_function_name=cost_function_name,
        env_params=args["env_params"],
    )

    submission_args = [
        f"experiment_setting={inputs.experiment_setting}",
        f"cost_function={inputs.cost_function}",
        f"file_number={inputs.file_number + 1}",
        f"reward_line={inputs.value_line}",
        f"bid={inputs.bid}",
    ]
    command = (
        f"condor_submit_bid {inputs.bid} "
        f"{Path(__file__).resolve().parents[2]}/cluster/submission_scripts/"
        f"MPI-IS/03_Get_Q_Values_broken_cluster.sub "
        f"{' '.join(submission_args)}"
    )
    os.system(command)
