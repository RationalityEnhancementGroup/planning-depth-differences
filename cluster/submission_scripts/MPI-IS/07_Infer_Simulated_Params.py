import os
from argparse import ArgumentParser
from pathlib import Path

if __name__ == "__main__":
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
        "-t",
        "--temperature-file",
        dest="temperature_file",
        help="File with temperatures to infer over",
        type=str,
        default=None,
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
        "-v",
        "--values",
        dest="cost_parameter_file",
        help="Cost parameter value file",
        type=str,
        default="params_full",
    )
    inputs = parser.parse_args()

    irl_folder = Path(__file__).resolve().parents[3]

    if inputs.policy == "RandomPolicy":
        inputs.simulated_cost_function = ""

    file_pattern = (
        f"cluster/data/trajectories/{inputs.experiment_setting}/"
        f"{inputs.policy}/simulated_agents_{inputs.simulated_cost_function}*"
    )

    for file in irl_folder.glob(file_pattern):
        submission_args = [
            f"sim_experiment_file={file}",
            f"cost_function={inputs.cost_function}",
            f"temperature_file={inputs.temperature_file}",
            f"param_file={inputs.cost_parameter_file}",
        ]
        command = (
            f"condor_submit_bid {inputs.bid} "
            f"{irl_folder}/cluster/"
            f"submission_scripts/MPI-IS/07_Infer_Simulated_Params.sub "
            f"{' '.join(submission_args)}"
        )
        os.system(command)
