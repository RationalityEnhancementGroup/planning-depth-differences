""" # noqa : D200
Script for investigating choosing parameters \
from full model vs choosing best model per participant.
"""
import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from costometer.utils import (
    AnalysisObject,
    get_correlation_text,
    get_static_palette,
    get_wilcoxon_text,
    set_font_sizes,
)

set_font_sizes()


def plot_bms_exceedance_probs(
    bms_out_df: pd.DataFrame,
    subdirectory,
    experiment_name,
    palette: Dict[str, Any] = None,
) -> None:
    """
    Plot BMS exceedance probabilities.

    :param bms_out_df: BMS results dataframe, including "Model" and \
        "Exceedance Probabilities"
    :param subdirectory:
    :param experiment_name:
    :param palette: palette as a dictionary models -> color
    :return: None
    """
    if palette is None:
        palette = get_static_palette(subdirectory, experiment_name)
    bar_order = (
        bms_out_df["Expected number of participants best explained by the model"]
        .sort_values()
        .index
    )
    plt.figure(figsize=(12, 8), dpi=80)
    sns.barplot(
        x="Exceedance Probabilities",
        y="Model",
        data=bms_out_df,
        palette=palette,
        order=bms_out_df.loc[bar_order, "Model"],
    )
    plt.title("Exceedance Probabilities")
    plt.xlabel("")


if __name__ == "__main__":
    """
    Example usage:
    python src/plot_bms.py -e MainExperiment
    """
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp",
        dest="experiment_name",
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
    optimization_data = optimization_data[
        optimization_data.apply(
            lambda row: set(analysis_obj.excluded_parameters.split(",")).issubset(
                row["model"]
            )
            or (row["Model Name"] == "Null"),
            axis=1,
        )
    ]

    irl_path.joinpath("data/bms/inputs/").mkdir(parents=True, exist_ok=True)
    irl_path.joinpath("data/bms/outputs/").mkdir(parents=True, exist_ok=True)

    if not irl_path.joinpath(f"data/bms/inputs/{inputs.experiment_name}.csv").is_file():
        pivoted_df = optimization_data.pivot(
            index="trace_pid", columns="Model Name", values="bic"
        )
        pivoted_df = pivoted_df.apply(lambda evidence: -0.5 * evidence)
        pivoted_df.to_csv(
            irl_path.joinpath(f"data/bms/inputs/{inputs.experiment_name}.csv")
        )
        quit()
    else:
        pivoted_df = pd.read_csv(
            irl_path.joinpath(f"data/bms/inputs/{inputs.experiment_name}.csv")
        )
        bms_df = pd.read_csv(
            irl_path.joinpath(f"data/bms/outputs/{inputs.experiment_name}.csv"),
            header=None,
        )

        pivoted_df = pivoted_df.set_index("trace_pid")
        bms_df.columns = pivoted_df.columns
        bms_df.index = pivoted_df.index

        results = []
        for row_idx, row in bms_df.iterrows():
            logging.info([row_idx, np.max(row), bms_df.columns[np.argmax(row)]])
            results.append([row_idx, np.max(row), bms_df.columns[np.argmax(row)]])

        new_bms_df = pd.DataFrame(results, columns=["pid", "prob_model", "model_name"])

        new_bms_df = new_bms_df.merge(
            optimization_data,
            left_on=["pid", "model_name"],
            right_on=["trace_pid", "Model Name"],
        )

        model_params = set(analysis_obj.cost_details["constant_values"]) - set(
            analysis_obj.excluded_parameters.split(",")
        )

        optimization_data = analysis_obj.query_optimization_data(
            excluded_parameters=analysis_obj.excluded_parameters
        )
        import pingouin as pg

        for param in model_params:
            logging.info("----------")
            logging.info(param)
            logging.info("----------")

            # This is the correlation between
            # fixed and random effect inferred parameters.
            correlation_object = pg.corr(
                optimization_data.sort_values(by="trace_pid")[param],
                new_bms_df.sort_values(by="pid")[param],
            )
            logging.info(get_correlation_text(correlation_object))

            # Non-parametric paired t-test for parameter values
            wilcoxon_object = pg.wilcoxon(
                optimization_data.sort_values(by="trace_pid")[param],
                new_bms_df.sort_values(by="pid")[param],
            )
            logging.info(get_wilcoxon_text(wilcoxon_object))

            plotting_data_fixed = (
                optimization_data.sort_values(by="trace_pid")[param]
                .copy(deep=True)
                .to_frame()
            )
            plotting_data_fixed["type"] = "fixed"

            plotting_data_random = (
                new_bms_df.sort_values(by="pid")[param].copy(deep=True).to_frame()
            )
            plotting_data_random["type"] = "random"

            plotting_data = pd.concat([plotting_data_fixed, plotting_data_random])

            plt.figure()
            sns.violinplot(data=plotting_data, x="type", y=param)
            plt.savefig(
                subdirectory.joinpath(
                    f"figs/{inputs.experiment_name}_{param}_fixed_vs_random.png"
                ),
                bbox_inches="tight",
            )
