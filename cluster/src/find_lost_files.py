"""
Finds Q values that did not complete on the cluster, and creates file for resubmitting
"""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from cluster_utils import get_args_from_yamls
from costometer.utils import get_param_string

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
        "-e",
        "--experiment",
        dest="experiment",
        help="Experiment (only used for finding missing likelihood files)",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-t",
        "--type",
        dest="q",
        help="If True, looks at Q files else looks at likelihood file",
        default=True,
        type=bool,
    )

    inputs = parser.parse_args()

    if not inputs.q:
        assert inputs.experiment is not None

    args = get_args_from_yamls(
        vars(inputs), attributes=["cost_function", "experiment_setting"]
    )

    # get path to save dictionary in
    cluster_path = Path(__file__).parents[1]

    full_parameters = np.loadtxt(
        cluster_path.joinpath(f"parameters/cost/{inputs.cost_parameter_file}.txt"),
        delimiter=",",
    )
    missing_parameters = []

    for curr_parameters in full_parameters:
        try:
            cost_parameters = {
                cost_parameter_arg: float(arg)
                for arg, cost_parameter_arg in zip(
                    curr_parameters, args["cost_parameter_args"]
                )
            }
        except ValueError as e:
            raise e

        parameter_string = get_param_string(cost_params=cost_parameters)

        if inputs.q:
            files = list(
                cluster_path.glob(
                    f"data/q_files/"
                    f"{inputs.experiment_setting}/{inputs.cost_function}/"
                    f"Q_{inputs.experiment_setting}_{parameter_string}_*.pickle"
                )
            )
        else:
            files = list(
                cluster_path.glob(
                    f"data/logliks/"
                    f"{inputs.cost_function}/{inputs.experiment}/"
                    f"SoftmaxPolicy_optimization_results_{parameter_string}.csv"
                )
            )

        if len(files) == 0:
            missing_parameters.append(curr_parameters)
        else:
            print(curr_parameters)
            print(f"Q_{inputs.experiment_setting}_{parameter_string}_*.pickle")

    np.savetxt(
        cluster_path.joinpath(
            f"parameters/cost/"
            f"{inputs.experiment_setting if inputs.q else inputs.experiment}"
            f"_{inputs.cost_function}_missing.txt"
        ),
        missing_parameters,
        fmt="%.02f",
        delimiter=",",
    )
