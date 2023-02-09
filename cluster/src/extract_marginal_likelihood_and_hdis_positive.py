from argparse import ArgumentParser
from pathlib import Path

import dill as pickle
import numpy as np
import pandas as pd
import yaml
from costometer.utils import (
    add_cost_priors_to_temp_priors,
    greedy_hdi_quantification,
    marginalize_out_for_data_set,
)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp",
        dest="experiment_name",
        metavar="experiment_name",
    )
    parser.add_argument(
        "-c",
        "--cost-function",
        dest="cost_function",
        type=str,
        default="dist_depth_eff_forw",
    )
    parser.add_argument(
        "-p",
        "--pid",
        dest="pid",
        type=int,
        default=None,
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
        "-x",
        "--excluded-params",
        dest="excluded_params",
        default="",
        type=str,
    )
    parser.add_argument(
        "-t",
        "--temperature-file",
        dest="temperature_file",
        help="File with temperatures to infer over",
        type=str,
        default="expon",
    )
    inputs = parser.parse_args()

    irl_path = Path(__file__).resolve().parents[2]
    irl_path.joinpath(
        f"cluster/data/marginal_hdi/{inputs.cost_function}/{inputs.experiment_name}/"
    ).mkdir(parents=True, exist_ok=True)

    # read in cost function details
    yaml_path = irl_path.joinpath(
        f"data/inputs/yamls/cost_functions/{inputs.cost_function}.yaml"
    )
    with open(yaml_path, "r") as stream:
        cost_details = yaml.safe_load(stream)

    temp_prior_details = {}
    for prior in inputs.temperature_file.split(","):
        yaml_path = irl_path.joinpath(f"data/inputs/yamls/temperatures/{prior}.yaml")
        with open(yaml_path, "r") as stream:
            prior_inputs = yaml.safe_load(stream)
        temp_prior_details[prior] = prior_inputs

    if inputs.excluded_params != "":
        file_end = "_" + inputs.excluded_params
    else:
        file_end = ""

    marginal_probability_file = irl_path.joinpath(
        f"cluster/data/marginal_hdi/"
        f"{inputs.cost_function}/{inputs.experiment_name}/"
        f"{inputs.block}_positive_map_{inputs.temperature_file}"
        f"_marginal_{inputs.pid:.0f}{file_end}.pickle"
    )
    if not marginal_probability_file.is_file():
        data = pd.concat(
            [
                pd.read_feather(
                    irl_path.joinpath(
                        f"cluster/data/logliks/{inputs.cost_function}/"
                        f"{inputs.experiment_name}"
                        f"{'_' + block if block != 'test' else ''}"
                        f"_by_pid/{inputs.pid}.feather"
                    )
                )
                for block in inputs.block.split(",")
            ]
        )
        data = data[(data["depth_cost_weight"] >= 0) & (data["given_cost"] >= 0)].copy(
            deep=True
        )

        prior_dict = {}
        yaml_path = irl_path.joinpath(
            f"data/inputs/yamls/temperatures/{inputs.temperature_file}.yaml"
        )
        with open(yaml_path, "r") as stream:
            prior_inputs = yaml.safe_load(stream)
        prior_dict[inputs.temperature_file] = prior_inputs

        prior_dict = add_cost_priors_to_temp_priors(
            data,
            cost_details,
            temp_prior_details,
            additional_params=["kappa", "gamma"],
        )[inputs.temperature_file]

        # exclude models where these params are varied
        subset = inputs.excluded_params.split(",")

        # subset data to only include where params of interest are varied
        data = data[(data["applied_policy"] == "SoftmaxPolicy")]

        if inputs.excluded_params != "":
            data = data[
                data.apply(
                    lambda row: sum(
                        row[cost_param] == cost_details["constant_values"][cost_param]
                        for cost_param in list(subset)
                    )
                    == len(list(subset)),
                    axis=1,
                )
            ]

        data = data.copy(deep=True).reset_index(drop=True)

        # make sure map is correct for our submodel
        data[f"map_{inputs.temperature_file}"] = data.apply(
            lambda row: row["mle"]
            + sum(
                [
                    np.log(prior_dict[param][row[param]])
                    for param in prior_dict.keys()
                    if param not in subset
                ]
            ),
            axis=1,
        )

        marginal_probabilities = marginalize_out_for_data_set(
            data=data,
            cost_parameter_args=cost_details["constant_values"].keys(),
            loglik_field=f"map_{inputs.temperature_file}",
        )
        with open(
            marginal_probability_file,
            "wb",
        ) as f:
            pickle.dump(marginal_probabilities, f)
    else:
        with open(
            marginal_probability_file,
            "rb",
        ) as f:
            marginal_probabilities = pickle.load(f)

    hdi_file = irl_path.joinpath(
        f"cluster/data/marginal_hdi/"
        f"{inputs.cost_function}/{inputs.experiment_name}/"
        f"{inputs.block}_positive_{inputs.temperature_file}"
        f"_hdi_{inputs.pid:.0f}{file_end}.pickle"
    )
    if not hdi_file.is_file():
        hdi_ranges = {}
        for parameter in cost_details["constant_values"]:
            param_df = pd.DataFrame(marginal_probabilities[parameter])

            sim_cols = [col for col in list(param_df) if "sim_" in str(col)]
            param_df = param_df.set_index(["trace_pid"] + sim_cols)

            for _, row in np.exp(param_df).iterrows():
                hdi_ranges[parameter] = greedy_hdi_quantification(
                    row, list(row.index.values)
                )

        with open(
            hdi_file,
            "wb",
        ) as f:
            pickle.dump(hdi_ranges, f)
