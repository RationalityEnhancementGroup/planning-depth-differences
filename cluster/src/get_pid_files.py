"""
Finds Q values that did not complete on the cluster, and creates file for resubmitting
"""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

if __name__ == "__main__":
    """
    Example call: python src/get_q_values.py -s experiment_setting -c cost \
    -v cost_parameter_values
    """
    # get arguments
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--experiment",
        dest="experiment",
        type=str,
    )

    inputs = parser.parse_args()

    # get path to save dictionary in
    irl_path = Path(__file__).resolve().parents[2]
    data = pd.read_csv(
        irl_path.joinpath(f"data/processed/{inputs.experiment}/mouselab-mdp.csv")
    )

    np.savetxt(
        irl_path.joinpath(f"cluster/parameters/pids/" f"{inputs.experiment}.txt"),
        data["pid"].unique(),
        fmt="%i",
        delimiter=",",
    )
