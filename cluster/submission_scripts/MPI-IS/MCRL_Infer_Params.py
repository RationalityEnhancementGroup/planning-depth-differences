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
        "--model",
        dest="model",
        help="Model row",
        type=str,
    )
    parser.add_argument(
        "-n",
        "--num-trials",
        dest="num_trials",
        help="Number of trials",
        type=str,
    )
    inputs = parser.parse_args()

    irl_folder = Path(__file__).resolve().parents[3]

    pid_range = np.unique(
        pd.read_csv(
            irl_folder.joinpath(f"data/processed/{inputs.experiment}/mouselab-mdp.csv")
        )["pid"]
    )

    irl_folder.joinpath("cluster/parameters/pids").mkdir(parents=True, exist_ok=True)
    np.savetxt(
        irl_folder.joinpath(f"cluster/parameters/pids/{inputs.experiment}.txt"),
        pid_range,
        fmt="%i",
    )

    submission_args = [
        f"experiment={inputs.experiment}",
        f"model={inputs.model}",
        f"num_trials={inputs.num_trials}",
    ]
    command = (
        f"condor_submit_bid {inputs.bid} "
        f"{irl_folder}/cluster/submission_scripts/"
        f"MPI-IS/MCRL_01_Infer_Params_by_PID.sub "
        f"{' '.join(submission_args)}"
    )
    os.system(command)
