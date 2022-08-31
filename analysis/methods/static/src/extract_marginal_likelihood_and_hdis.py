from argparse import ArgumentParser
from pathlib import Path

import dill as pickle
import numpy as np
import pandas as pd
from costometer.utils import (
    AnalysisObject,
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
        "-s",
        "--subdirectory",
        dest="experiment_subdirectory",
        metavar="experiment_subdirectory",
    )
    parser.add_argument(
        "-c",
        "--cost-function",
        dest="cost_function",
        type=str,
        default="linear_depth",
    )
    parser.add_argument(
        "-t",
        "--simulated-temperature",
        dest="simulated_temperature",
        type=float,
        default=None,
    )
    parser.add_argument(
        "-p",
        "--pid",
        dest="pid",
        type=int,
        default=None,
    )
    inputs = parser.parse_args()

    data_path = Path(__file__).resolve().parents[1]
    irl_path = Path(__file__).resolve().parents[4]
    irl_path.joinpath(
        f"analysis/{inputs.experiment_subdirectory}/" f"data/{inputs.experiment_name}/"
    ).mkdir(parents=True, exist_ok=True)

    analysis_obj = AnalysisObject(
        inputs.experiment_name,
        irl_path=irl_path,
        experiment_subdirectory=inputs.experiment_subdirectory,
    )

    if inputs.pid is not None:
        pid_text = f"_{inputs.pid:.0f}"
    else:
        pid_text = ""

    if inputs.simulated_temperature is not None:
        temp_text = f"_{inputs.simulated_temperature:.2f}"
    else:
        temp_text = ""

    full_marginal_probabilities = {}
    for block in analysis_obj.block:
        marginal_probability_file = irl_path.joinpath(
            f"analysis/{inputs.experiment_subdirectory}/"
            f"data/{inputs.experiment_name}/{inputs.experiment_name}"
            f"_{inputs.cost_function}_{block}_map_{analysis_obj.prior}"
            f"_marginal{pid_text}{temp_text}.pickle"
        )
        print(marginal_probability_file)
        if not marginal_probability_file.is_file():
            if not analysis_obj.simulated:
                data = pd.concat(
                    [
                        pd.read_feather(
                            irl_path.joinpath(
                                f"cluster/data/logliks/{inputs.cost_function}/"
                                f"{session}.feather"
                            )
                        )
                        for session in analysis_obj.sessions
                    ]
                )
            else:
                data = pd.concat(
                    [
                        pd.read_feather(f)
                        for session in analysis_obj.sessions
                        for f in irl_path.glob(
                            f"cluster/data/logliks/{inputs.cost_function}/"
                            f"simulated/{analysis_obj.experiment_setting}/"
                            f"{session.split('/')[0]}*_applied{temp_text}.feather"
                        )
                    ]
                )

            if inputs.simulated_temperature is not None:
                data = data[data["sim_temp"] == inputs.simulated_temperature]
            if inputs.pid is not None:
                data = data[data["trace_pid"] == inputs.pid]

            marginal_probabilities = marginalize_out_for_data_set(
                data[(data["applied_policy"] == "SoftmaxPolicy")],
                cost_parameter_args=analysis_obj.cost_details[inputs.cost_function][
                    "cost_parameter_args"
                ],
                loglik_field=f"{block}_map_{analysis_obj.prior}"
                if block != "All"
                else f"map_{analysis_obj.prior}",
            )
            full_marginal_probabilities[block] = marginal_probabilities
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
            full_marginal_probabilities[block] = marginal_probabilities

    hdi_file = irl_path.joinpath(
        f"analysis/{inputs.experiment_subdirectory}/"
        f"data/{inputs.experiment_name}/{inputs.experiment_name}_{inputs.cost_function}"
        f"_hdi{pid_text}{temp_text}.pickle"
    )

    if not hdi_file.is_file():
        hdi_ranges = {
            block: {
                parameter: {}
                for parameter in analysis_obj.cost_details[inputs.cost_function][
                    "cost_parameter_args"
                ]
                + ["temp"]
            }
            for block in full_marginal_probabilities.keys()
        }
        for block in full_marginal_probabilities.keys():
            for parameter in analysis_obj.cost_details[inputs.cost_function][
                "cost_parameter_args"
            ] + ["temp"]:
                param_df = pd.DataFrame(full_marginal_probabilities[block][parameter])
                if not analysis_obj.simulated:
                    param_df = param_df.set_index(["trace_pid"])
                else:
                    sim_cols = [col for col in list(param_df) if "sim_" in str(col)]
                    param_df = param_df.set_index(["trace_pid"] + sim_cols)

                for pid, probs in np.exp(param_df).iterrows():
                    hdi_ranges[block][parameter][pid] = greedy_hdi_quantification(
                        probs, list(probs.index.values)
                    )

        with open(
            hdi_file,
            "wb",
        ) as f:
            pickle.dump(hdi_ranges, f)
