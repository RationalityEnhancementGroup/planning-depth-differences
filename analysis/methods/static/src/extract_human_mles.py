from argparse import ArgumentParser
from pathlib import Path

import dill as pickle
import pandas as pd
import yaml
from costometer.utils import add_cost_priors_to_temp_priors, extract_mles_and_maps
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
        "-d",
        "--participant-subset-file",
        dest="participant_subset_file",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-k",
        "--block",
        dest="block",
        default=None,
        help="Block",
        type=str,
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
        "-p",
        "--pid",
        dest="pid",
        help="Participant ID (optional)",
        default=None,
    )

    inputs = parser.parse_args()
    irl_path = Path(__file__).resolve().parents[4]
    data_path = Path(__file__).resolve().parents[1]

    # get priors

    temp_prior_details = {}
    for prior in inputs.temperature_file.split(","):
        yaml_path = irl_path.joinpath(f"data/inputs/yamls/temperatures/{prior}.yaml")
        with open(yaml_path, "r") as stream:
            prior_inputs = yaml.safe_load(stream)
        temp_prior_details[prior] = prior_inputs

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

    if inputs.block != "test":
        simulation_params = "_" + inputs.block
    else:
        simulation_params = ""
    
    if inputs.pid:
        data = pd.read_feather(
            irl_path.joinpath(
                f"cluster/data/logliks/{inputs.cost_function}/"
                f"{inputs.experiment}{simulation_params}_by_pid/{inputs.pid}.feather"
            )
        )
    else:
        data = pd.read_feather(
            irl_path.joinpath(
                f"cluster/data/logliks/{inputs.cost_function}/"
                f"{inputs.experiment}{simulation_params}.feather"
            )
        )

    full_priors = add_cost_priors_to_temp_priors(
        data,
        cost_details,
        temp_prior_details,
        additional_params=["kappa", "gamma"],
    )

    best_parameter_values = extract_mles_and_maps(data, cost_details, full_priors)

    # create cost subfolder if not already there
    irl_path.joinpath(
        f"data/processed/{inputs.experiment}"
        f"{simulation_params}/{inputs.cost_function}"
    ).mkdir(parents=True, exist_ok=True)
    if inputs.pid:
        with open(
            irl_path.joinpath(
                f"data/processed/{inputs.experiment}"
                f"/{inputs.cost_function}/"
                f"mle_and_map{simulation_params}"
                f"_{inputs.pid}.pickle"
            ),
            "wb",
        ) as f:
            pickle.dump(best_parameter_values, f)
    else:
        with open(
            irl_path.joinpath(
                f"data/processed/{inputs.experiment}"
                f"/{inputs.cost_function}/"
                f"mle_and_map{simulation_params}.pickle"
            ),
            "wb",
        ) as f:
            pickle.dump(best_parameter_values, f)
