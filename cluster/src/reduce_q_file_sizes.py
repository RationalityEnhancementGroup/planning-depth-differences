import os
from argparse import ArgumentParser
from pathlib import Path

import blosc
import dill as pickle

if __name__ == "__main__":
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
    inputs = parser.parse_args()

    for curr_file in (
        Path(__file__)
        .parents[1]
        .glob(
            f"data/q_files/{inputs.experiment_setting}/{inputs.cost_function}/Q*.pickle"
        )
    ):
        with open(curr_file, "rb") as f:
            data = pickle.load(f)

        pickled_data = pickle.dumps(data)
        compressed_pickle = blosc.compress(pickled_data)

        with open(curr_file.parent.joinpath(f"{curr_file.stem}.dat"), "wb") as f:
            f.write(compressed_pickle)
        os.remove(curr_file)
