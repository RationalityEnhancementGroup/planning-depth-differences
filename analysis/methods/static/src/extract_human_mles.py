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
    parser.add_argument(
        "-s",
        "--simulated-cost-function",
        dest="simulated_cost_function",
        help="Simulated cost function YAML file",
        default="back_dist_depth_eff_forw",
        type=str,
    )
    parser.add_argument(
        "-p",
        "--pid",
        dest="pid",
        help="Participant ID (optional)",
        default=None,
    )
    parser.add_argument(
        "-a",
        "--alpha",
        dest="alpha",
        help="alpha",
        type=float,
        default=1,
    )
    parser.add_argument(
        "-g",
        "--gamma",
        dest="gamma",
        help="gamma",
        type=float,
        default=1,
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
        f"data/inputs/yamls/cost_functions/{inputs.simulated_cost_function}.yaml"
    )

    with open(yaml_path, "r") as stream:
        cost_details = yaml.safe_load(stream)

    if inputs.alpha == 1:
        alpha_string = ""
    else:
        alpha_string = f"_{inputs.alpha:.2f}"

    if inputs.gamma == 1:
        gamma_string = ""
    else:
        gamma_string = f"{inputs.gamma:.3f}"

    if inputs.pid:
        data = pd.read_feather(
            irl_path.joinpath(
                f"cluster/data/logliks/{inputs.cost_function}/"
                f"{inputs.experiment}{gamma_string}{alpha_string}_by_pid/{inputs.pid}.feather"
            )
        )
    else:
        data = pd.read_feather(
            irl_path.joinpath(
                f"cluster/data/logliks/{inputs.cost_function}/"
                f"{inputs.experiment}{gamma_string}{alpha_string}.feather"
            )
        )

    priors = pickle.load(
        open(
            irl_path.joinpath(
                f"cluster/data/priors/"
                f"{inputs.cost_function}/"
                f"{inputs.experiment}{gamma_string}{alpha_string}.pkl"
            ),
            "rb",
        ),
    )

    best_parameter_values = extract_mles_and_maps(data, cost_details, priors)

    # create cost subfolder if not already there
    irl_path.joinpath(
        f"data/processed/{inputs.experiment}{gamma_string}"
        f"{alpha_string}/{inputs.cost_function}"
    ).mkdir(parents=True, exist_ok=True)
    if inputs.pid:
        with open(
            irl_path.joinpath(
                f"data/processed/{inputs.experiment}{gamma_string}"
                f"{alpha_string}/{inputs.cost_function}/"
                f"mle_and_map_{inputs.pid}.pickle"
            ),
            "wb",
        ) as f:
            pickle.dump(best_parameter_values, f)
    else:
        with open(
            irl_path.joinpath(
                f"data/processed/{inputs.experiment}{gamma_string}"
                f"{alpha_string}/{inputs.cost_function}/"
                f"mle_and_map.pickle"
            ),
            "wb",
        ) as f:
            pickle.dump(best_parameter_values, f)
