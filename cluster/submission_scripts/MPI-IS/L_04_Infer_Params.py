import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

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
        "-e",
        "--experiment",
        dest="experiment",
        help="Experiment",
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
        type=str,
        default="search_space",
    )
    parser.add_argument(
        "-o",
        "--constant-yaml",
        dest="constant_yaml",
        help="Constant YAML",
        type=str,
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
        dest="cost_parameter_file",
        help="Cost parameter value file",
        type=str,
        default="params_full",
    )
    inputs = parser.parse_args()

    irl_folder = Path(__file__).resolve().parents[3]

    pid_range = np.unique(
        pd.read_csv(
            irl_folder.joinpath(f"data/processed/{inputs.experiment}/mouselab-mdp.csv")
        )["pid"]
    )

    for pid in pid_range:
        submission_args = [
            f"experiment={inputs.experiment}",
            f"model={inputs.model_yaml}",
            f"features={inputs.feature_yaml}",
            f"pid={pid}",
            f"param_file={inputs.cost_parameter_file}",
            f"priors={inputs.prior_json}",
            f"constant_params={inputs.constant_yaml}",
            f"cost_function={inputs.cost_function}",
        ]
        command = (
            f"condor_submit_bid {inputs.bid} "
            f"L_04_Infer_Params_by_PID.sub "
            f"{' '.join(submission_args)}"
        )
        os.system(command)
