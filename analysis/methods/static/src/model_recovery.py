from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from costometer.utils import AnalysisObject, get_static_palette, set_font_sizes
from statsmodels.tools.eval_measures import bic

set_font_sizes(SMALL_SIZE=14)

###################################################
# This section contains my plotting functions
###################################################


def bic_plot(
    optimization_data, subdirectory, experiment_name, bic_field="bic", palette=None
):
    if palette is None:
        palette = get_static_palette(subdirectory, experiment_name)
    plt.figure(figsize=(12, 8), dpi=80)

    sns.barplot(
        y="Model Name",
        x=bic_field,
        data=optimization_data.sort_values(by="bic")[:33],
        palette=palette,
    )

    plt.xlabel(None)
    plt.ylabel(None)

    plt.gca().set_xlim((0.75 * min(optimization_data["bic"]), plt.gca().get_xlim()[1]))


if __name__ == "__main__":
    """
    Example usage:
    python src/plot_bic.py -e MainExperiment
    """
    parser = ArgumentParser()
    parser.add_argument(
        "-e", "--exp", dest="experiment_name", default="SoftmaxRecovery"
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
    subdirectory = irl_path.joinpath(f"analysis/{inputs.experiment_subdirectory}")

    analysis_obj = AnalysisObject(
        inputs.experiment_name,
        irl_path=irl_path,
        experiment_subdirectory=inputs.experiment_subdirectory,
    )

    optimization_data = analysis_obj.query_optimization_data()

    if analysis_obj.excluded_parameters == "":
        excluded_set = set()
    else:
        excluded_set = set(analysis_obj.excluded_parameters.split(","))

    optimization_data = optimization_data[
        optimization_data.apply(
            lambda row: excluded_set.issubset(row["model"])
            or (row["Model Name"] == "Null"),
            axis=1,
        )
    ]

    optimization_data = optimization_data[
        optimization_data["applied_policy"] != "RandomPolicy"
    ]
    # unfortunately don't have given cost 1
    analysis_obj.cost_details["constant_values"]["given_cost"] = 0

    full_df = []
    for model in (
        optimization_data["model"]
        .apply(lambda model: tuple(model) if model != "None" else ())
        .unique()
    ):
        if model != "":
            curr_optimization_data = optimization_data[
                optimization_data.apply(
                    lambda row: np.all(
                        [
                            row[f"sim_{param}"]
                            == analysis_obj.cost_details["constant_values"][param]
                            for param in model
                        ]
                    ),
                    axis=1,
                )
            ]
        else:
            curr_optimization_data = optimization_data

        bic_df = (
            curr_optimization_data.groupby(["Model Name", "Number Parameters"])
            .sum()
            .reset_index()
        )

        bic_df["bic"] = bic_df.apply(
            lambda row: bic(
                llf=row["mle"],
                nobs=row["num_clicks"],
                df_modelwc=row["Number Parameters"],
            ),
            axis=1,
        )

        curr_df = bic_df[["Model Name", "bic"]]
        curr_df["Simulated Model"] = analysis_obj.model_name_mapping[model]
        full_df.append(curr_df)

    full_df = pd.concat(full_df)

    full_df = full_df[
        full_df.apply(
            lambda row: row["Model Name"] in full_df["Simulated Model"].unique(), axis=1
        )
    ]
    heat_map_data = full_df.pivot(
        index="Model Name", columns="Simulated Model", values="bic"
    )
    for col in heat_map_data.columns:
        heat_map_data[col] = heat_map_data[col] - min(heat_map_data[col])

    plt.figure(figsize=(80, 60))
    sns.heatmap(data=heat_map_data, annot=True, fmt=".2f")

    plt.savefig(
        subdirectory.joinpath(f"figs/{inputs.experiment_name}_model_recovery.png"),
        bbox_inches="tight",
    )
