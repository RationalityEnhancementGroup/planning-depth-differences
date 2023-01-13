from argparse import ArgumentParser
from pathlib import Path

import dill as pickle
import numpy as np
import pandas as pd
import yaml
from costometer.utils import extract_mles_and_maps, get_param_string
from scipy import stats  # noqa
import os
from costometer.utils import get_param_string

if __name__ == "__main__":
    """
    python extract_simulated_mles.py -p OptimalQ -c dist_depth_eff_forw -o \
    dist_depth_eff_forw -e high_increasing
    python extract_simulated_mles.py -p SoftmaxPolicy -c dist_depth_eff_forw -o \
    dist_depth_eff_forw -e high_increasing
    python extract_simulated_mles.py -p RandomPolicy -c dist_depth_eff_forw -o \
    dist_depth_eff_forw -e high_increasing
    """
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
        "--simulated-temperature",
        dest="simulated_temperature",
        type=float,
        default=None,
    )
    parser.add_argument(
        "-a",
        "--values",
        dest="cost_parameter_values",
        default=None,
        help="Cost parameter values as comma separated string, e.g. '1.00,2.00'",
        type=str,
    )

    inputs = parser.parse_args()

    irl_path = Path(__file__).resolve().parents[4]
    data_path = Path(__file__).resolve().parents[1]

    # we load and save files by temperature, if simulated temp provided
    if inputs.simulated_temperature is not None:
        temp_string = f"_{inputs.simulated_temperature:.2f}"
    else:
        temp_string = ""

    if inputs.policy == "RandomPolicy":
        inputs.simulated_cost_function = ""

    if inputs.cost_parameter_values:
        yaml_path = irl_path.joinpath(
            f"data/inputs/yamls/cost_functions/{inputs.cost_function}.yaml"
        )
        with open(yaml_path, "r") as stream:
            cost_details = yaml.safe_load(stream)
        param_string = get_param_string({cost_parameter_arg: arg for arg, cost_parameter_arg in zip( inputs.cost_parameter_values.split(","), cost_details["cost_parameter_args"])})
        
        data = pd.read_feather(
            irl_path.joinpath(
                f"cluster/data/logliks/{inputs.cost_function}/simulated/"
                f"{inputs.experiment_setting}/"
                f"{inputs.policy}_by_pid/"
                f"{inputs.policy}"
                f"{'_' if len(inputs.simulated_cost_function) > 0 else ''}"
                f"{inputs.simulated_cost_function}_applied_{param_string}{temp_string}.feather"
            )
        )
    else:
        data = pd.read_feather(
            irl_path.joinpath(
                f"cluster/data/logliks/{inputs.cost_function}/simulated/"
                f"{inputs.experiment_setting}/{inputs.policy}"
                f"{'_' if len(inputs.simulated_cost_function) > 0 else ''}"
                f"{inputs.simulated_cost_function}_applied{temp_string}.feather"
            )
        )

    if inputs.simulated_temperature is not None:
        data = data[data["sim_temp"] == inputs.simulated_temperature].reset_index()

    yaml_path = irl_path.joinpath(
        f"data/inputs/yamls/cost_functions/{inputs.cost_function}.yaml"
    )
    with open(yaml_path, "r") as stream:
        cost_details = yaml.safe_load(stream)

    if inputs.cost_parameter_values is not None:
        cost_parameters = {
            cost_parameter_arg: float(arg)
            for arg, cost_parameter_arg in zip(
                inputs.cost_parameter_values.split(","),
                cost_details["cost_parameter_args"],
            )
        }
        data = data[
            data.apply(
                lambda row: np.all(
                    [
                        row[f"sim_{cost_parameter}"] == param_value
                        for cost_parameter, param_value in cost_parameters.items()
                    ]
                ),
                axis=1,
            )
        ]
        cost_string = f"_{get_param_string(cost_parameters)}"
    else:
        cost_string = ""

    priors = pickle.load(
        open(
            irl_path.joinpath(
                f"cluster/data/priors/"
                f"{inputs.cost_function}/"
                f"{inputs.policy}{'_' if len(inputs.simulated_cost_function) > 0 else ''}"
                f"{inputs.simulated_cost_function}_applied.pkl"
            ),
            "rb",
        ),
    )
    best_parameter_values = extract_mles_and_maps(data, cost_details, priors)
    
    # create cost subfolder if not already there
    irl_path.joinpath(
        f"data/processed/{inputs.policy}/{inputs.experiment_setting}/"
        f"{inputs.cost_function}"
    ).mkdir(parents=True, exist_ok=True)
    with open(
        irl_path.joinpath(
            f"data/processed/{inputs.policy}/{inputs.experiment_setting}/"
            f"{inputs.cost_function}/mle_and_map{cost_string}{temp_string}.pickle"
        ),
        "wb",
    ) as f:
        pickle.dump(best_parameter_values, f)

    #temps = ["1.00", "2.50", "7.50", "100.0", "0.50"]
    #curr_temp_index = [float(t) for t in temps].index(inputs.simulated_temperature)
    #next_temp_index = curr_temp_index+1
    #if next_temp_index < len(temps):
    #    command = (
    #        f"condor_submit_bid {inputs.bid} "
    #        f"{irl_path}/cluster/submission_scripts/"
    #        f"MPI-IS/M_01_Get_MAP_Simulated_by_Temp_broken_cluster.sub "
    #        f"temp={temps[next_temp_index]} "
    #        f"reward_line={inputs.cost_parameter_values}"
    #    )
    #    os.system(command)
