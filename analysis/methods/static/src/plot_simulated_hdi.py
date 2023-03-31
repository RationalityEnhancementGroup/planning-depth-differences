from argparse import ArgumentParser
from pathlib import Path

import numpy as np
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
        default="methods/static",
        dest="experiment_subdirectory",
        metavar="experiment_subdirectory",
    )
    inputs = parser.parse_args()

    irl_path = Path(__file__).resolve().parents[4]
    data_path = irl_path.joinpath(f"analysis/{inputs.experiment_subdirectory}")

    analysis_obj = AnalysisObject(
        inputs.experiment_name,
        irl_path=irl_path,
        experiment_subdirectory=inputs.experiment_subdirectory,
    )
    optimization_data = analysis_obj.query_optimization_data(
        excluded_parameters=analysis_obj.excluded_parameters
    )

    model_params = set(analysis_obj.cost_details["constant_values"]) - set(
        analysis_obj.excluded_parameters.split(",")
    )

    hdi_ranges = analysis_obj.load_hdi_ranges(
        excluded_parameters=analysis_obj.excluded_parameters
    )

    for parameter in model_params:
        optimization_data[f"min_{parameter}"] = optimization_data["trace_pid"].apply(
            lambda pid: hdi_ranges[pid][parameter][0]
        )
        optimization_data[f"max_{parameter}"] = optimization_data["trace_pid"].apply(
            lambda pid: hdi_ranges[pid][parameter][1]
        )

    for parameter in model_params:
        optimization_data[f"{parameter}_in"] = optimization_data.apply(
            lambda row: (row[f"sim_{parameter}"] <= row[f"max_{parameter}"])
            and (row[f"sim_{parameter}"] >= row[f"min_{parameter}"]),
            axis=1,
        )
        optimization_data[f"{parameter}_spread"] = optimization_data.apply(
            lambda row: row[f"max_{parameter}"] - row[f"min_{parameter}"], axis=1
        )

    print("Statistics for spread of parameters")
    for parameter in model_params:
        print(
            f"{analysis_obj.cost_details['latex_mapping'][parameter]}"
            f" & ${optimization_data[f'{parameter}_spread'].mean():.2f}$"
            f" (${optimization_data[f'{parameter}_spread'].std():.2f}$)"
        )

    # for cases where we don't vary temperature
    for parameter in model_params:
        print("----------")
        print(f"Correlation between spread of {parameter} and temperature")
        print("----------")
        correlation_object = pg.corr(
            optimization_data["sim_temp"], optimization_data[f"{parameter}_spread"]
        )
        print(get_correlation_text(correlation_object))

    for parameter in model_params:
        print("----------")
        print(f"Amount of time true {parameter} is in the outputted interval")
        print("----------")
        print(f"{optimization_data[f'{parameter}_in'].mean():.2f}")

    for parameter in model_params:
        optimization_data[f"diff_{parameter}"] = optimization_data.apply(
            lambda row: np.sqrt((row[parameter] - row[f"sim_{parameter}"]) ** 2),
            axis=1,
        )

        if len(optimization_data[f"{parameter}_in"].unique()) == 1:
            print("----------")
            print(
                f"True {parameter} value is always in or outside of outputted interval:"
            )
            print(optimization_data[f"{parameter}_in"].unique())
        else:
            print("----------")
            print(
                f"Difference between error when true {parameter} "
                f"parameter value is in outputted interval vs not."
            )
            print("----------")
            comparison = pg.mwu(
                optimization_data[optimization_data[f"{parameter}_in"]][
                    f"diff_{parameter}"
                ],
                optimization_data[~optimization_data[f"{parameter}_in"]][
                    f"diff_{parameter}"
                ],
            )
            print(get_mann_whitney_text(comparison))

    print("----------")
    print("Correlation between error in MAP estimate and spread")
    print("----------")
    for parameter in model_params:
        correlation_object = pg.corr(
            optimization_data[f"{parameter}_spread"],
            optimization_data[f"diff_{parameter}"],
        )
        print(
            f"{analysis_obj.cost_details['latex_mapping'][parameter]}"
            f" & {get_correlation_text(correlation_object, table=True)}"
        )

    print("----------")
    print("Correlation between MAP estimate and spread")
    print("----------")
    for parameter in model_params:
        correlation_object = pg.corr(
            optimization_data[parameter],
            optimization_data[f"{parameter}_spread"],
        )
        print(
            f"{analysis_obj.cost_details['latex_mapping'][parameter]}"
            f" & {get_correlation_text(correlation_object, table=True)}"
        )

    print("----------")
    print("Correlation between true value and spread")
    print("----------")
    for parameter in model_params:
        correlation_object = pg.corr(
            optimization_data[f"sim_{parameter}"],
            optimization_data[f"{parameter}_spread"],
        )
        print(
            f"{analysis_obj.cost_details['latex_mapping'][parameter]}"
            f" & {get_correlation_text(correlation_object, table=True)}"
        )
