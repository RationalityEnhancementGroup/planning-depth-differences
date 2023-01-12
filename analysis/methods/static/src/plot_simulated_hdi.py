from argparse import ArgumentParser
from pathlib import Path

import dill as pickle
import numpy as np
import pandas as pd
import pingouin as pg
from costometer.utils import (
    AnalysisObject,
    get_correlation_text,
    get_mann_whitney_text,
    set_font_sizes,
)

set_font_sizes()

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
        default="dist_depth_eff_forw",
    )
    inputs = parser.parse_args()

    irl_path = Path(__file__).resolve().parents[4]
    data_path = irl_path.joinpath(f"analysis/{inputs.experiment_subdirectory}")

    analysis_obj = AnalysisObject(
        inputs.experiment_name,
        irl_path=irl_path,
        experiment_subdirectory=inputs.experiment_subdirectory,
    )
    optimization_data = analysis_obj.query_optimization_data()
    optimization_data = optimization_data[
        optimization_data["Model Name"] == "'Distance, Effort, Depth and Forward Search Bonus'"
    ]

    if "sim_temp" in optimization_data:
        model_params = analysis_obj.cost_details[inputs.cost_function][
            "cost_parameter_args"
        ] + ["temp"]

        index_names = [
            "trace_pid",
            "sim_policy",
            "sim_experiment_setting",
            "sim_cost_function",
            "sim_temperature_file",
            "sim_cost_parameter_values",
            "sim_num_simulated",
            "sim_num_trials",
            "sim_seed",
            "sim_static_cost_weight",
            "sim_depth_cost_weight",
            "sim_temp",
            "sim_noise",
        ]

    else:
        model_params = analysis_obj.cost_details[inputs.cost_function][
            "cost_parameter_args"
        ]

        index_names = [
            "trace_pid",
            "sim_policy",
            "sim_experiment_setting",
            "sim_cost_function",
            "sim_temperature_file",
            "sim_cost_parameter_values",
            "sim_num_simulated",
            "sim_num_trials",
            "sim_seed",
            "sim_static_cost_weight",
            "sim_depth_cost_weight",
        ]

    optimization_data.set_index(index_names, inplace=True)

    hdi_files_names = []
    for temp in analysis_obj.temp:
        temp_text = f"_{temp:.2f}"
        hdi_files_names.extend(
            list(
                data_path.glob(
                    f"data/{inputs.experiment_name}/"
                    f"{inputs.experiment_name}_{inputs.cost_function}"
                    f"_hdi*{temp_text}.pickle"
                )
            )
        )

    hdi_dfs = []
    for hdi_file_name in hdi_files_names:
        with open(
            hdi_file_name,
            "rb",
        ) as f:
            hdi_ranges = pickle.load(f)
            hdi_dfs.append(pd.DataFrame.from_dict(hdi_ranges["All"]))

    hdi_df = pd.concat(hdi_dfs)
    hdi_df.index.set_names(index_names, inplace=True)

    for parameter in model_params:
        hdi_df[f"sim_{parameter}"] = hdi_df.index.get_level_values(f"sim_{parameter}")
        hdi_df[f"{parameter}_in"] = hdi_df.apply(
            lambda row: (row[f"sim_{parameter}"] <= row[parameter][1])
            and (row[f"sim_{parameter}"] >= row[parameter][0]),
            axis=1,
        )
        hdi_df[f"{parameter}_spread"] = hdi_df.apply(
            lambda row: row[parameter][1] - row[parameter][0], axis=1
        )

    for parameter in model_params:
        print("----------")
        print(f"Statistics for spread of parameter: {parameter}")
        print("----------")
        print(
            f"$M: {hdi_df[f'{parameter}_spread'].mean():.2f}, "
            f"SD: {hdi_df[f'{parameter}_spread'].std():.2f}$"
        )

    for parameter in model_params:
        print("----------")
        print(f"Correlation between spread of {parameter} and temperature")
        print("----------")
        correlation_object = pg.corr(hdi_df["sim_temp"], hdi_df[f"{parameter}_spread"])
        print(get_correlation_text(correlation_object))

    for parameter in model_params:
        print("----------")
        print(f"Amount of time true {parameter} is in the outputted interval")
        print("----------")
        print(f"{hdi_df[f'{parameter}_in'].mean():.2f}")

    full_df = hdi_df.join(optimization_data, lsuffix="", rsuffix="_map", how="left")

    print(full_df[f"{'temp'}_in"].unique())
    for parameter in model_params:
        full_df[f"diff_{parameter}"] = full_df.apply(
            lambda row: np.sqrt(
                (row[f"{parameter}_map"] - row[f"sim_{parameter}"]) ** 2
            ),
            axis=1,
        )

        if len(full_df[f"{parameter}_in"].unique()) == 1:
            print("----------")
            print(
                f"True {parameter} value is always in or outside of outputted interval:"
            )
            print(full_df[f"{parameter}_in"].unique())
        else:
            print("----------")
            print(
                f"Difference between error when true {parameter} "
                f"parameter value is in outputted interval vs not."
            )
            print("----------")
            comparison = pg.mwu(
                full_df[full_df[f"{parameter}_in"] == True][  # noqa: E712
                    f"diff_{parameter}"
                ],
                full_df[full_df[f"{parameter}_in"] == False][f"diff_{parameter}"],
            )
            print(get_mann_whitney_text(comparison))

        print("----------")
        print(f"Correlation between error in MAP estimate and spread for {parameter}")
        print("----------")
        correlation_object = pg.corr(
            full_df[f"{parameter}_spread"],
            full_df[f"diff_{parameter}"],
        )
        print(get_correlation_text(correlation_object))
        print("----------")
        print(f"Correlation between MAP estimate and spread for {parameter}")
        print("----------")
        correlation_object = pg.corr(
            full_df[f"{parameter}_map"],
            full_df[f"{parameter}_spread"],
        )
        print(get_correlation_text(correlation_object))
        print("----------")
        print(f"Correlation between true value and spread for {parameter}")
        print("----------")
        correlation_object = pg.corr(
            full_df[f"sim_{parameter}"],
            full_df[f"{parameter}_spread"],
        )
        print(get_correlation_text(correlation_object))
