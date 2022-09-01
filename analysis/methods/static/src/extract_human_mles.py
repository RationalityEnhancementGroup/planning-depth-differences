from argparse import ArgumentParser
from pathlib import Path

import dill as pickle
import pandas as pd
import yaml
from costometer.utils import extract_mles_and_maps
from scipy import stats  # noqa

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp",
        dest="experiment",
        metavar="experiment_name",
    )
    parser.add_argument(
        "-c",
        "--cost",
        dest="cost_function",
        help="Cost Function",
        metavar="cost_function",
    )

    inputs = parser.parse_args()
    irl_path = Path(__file__).resolve().parents[4]
    data_path = Path(__file__).resolve().parents[1]

    # read in experiment file
    yaml_path = irl_path.joinpath(
        f"data/inputs/yamls/experiments/{inputs.experiment}.yaml"
    )

    with open(yaml_path, "r") as stream:
        experiment_details = yaml.safe_load(stream)

    # read in cost function details
    yaml_path = irl_path.joinpath(
        f"data/inputs/yamls/cost_functions/{inputs.cost_function}.yaml"
    )

    with open(yaml_path, "r") as stream:
        cost_details = yaml.safe_load(stream)

    data = pd.read_feather(
        irl_path.joinpath(
            f"cluster/data/logliks/{inputs.cost_function}/"
            f"{inputs.experiment}.feather"
        )
    )

    priors = pickle.load(
        open(
            irl_path.joinpath(
                f"cluster/data/priors/"
                f"{inputs.cost_function}/"
                f"{inputs.experiment}.pkl"
            ),
            "rb",
        ),
    )

    best_parameter_values = extract_mles_and_maps(data, cost_details, priors)

    # create cost subfolder if not already there
    irl_path.joinpath(
        f"data/processed/{inputs.experiment}/{inputs.cost_function}"
    ).mkdir(parents=True, exist_ok=True)
    with open(
        irl_path.joinpath(
            f"data/processed/{inputs.experiment}/{inputs.cost_function}/"
            f"mle_and_map.pickle"
        ),
        "wb",
    ) as f:
        pickle.dump(best_parameter_values, f)
