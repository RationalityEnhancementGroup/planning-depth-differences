""" # noqa : D200
Script for investigating choosing parameters \
from full model vs choosing best model per participant.
"""
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from costometer.utils import get_correlation_text, get_static_palette, get_wilcoxon_text
from costometer.utils.scripting_utils import standard_parse_args


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
    irl_path = Path(__file__).resolve().parents[4]
    analysis_obj, inputs, subdirectory = standard_parse_args(
        description=sys.modules[__name__].__doc__,
        irl_path=irl_path,
        filename=Path(__file__).stem,
    )

    optimization_data = analysis_obj.query_optimization_data()
    optimization_data = optimization_data[
        optimization_data.apply(
            lambda row: set(analysis_obj.analysis_details.excluded_parameters).issubset(
                row["model"]
            )
            or (row["Model Name"] == "Null"),
            axis=1,
        )
    ]

    pivoted_df = pd.read_csv(
        irl_path.joinpath(f"data/bms/inputs/{inputs.experiment_name}.csv")
    )
    bms_df = pd.read_csv(
        irl_path.joinpath(f"data/bms/outputs/{inputs.experiment_name}.csv"),
        header=None,
    )

    bms_df.columns = pivoted_df.columns
    pivoted_df = pivoted_df.set_index("trace_pid")
    bms_df = bms_df.set_index("trace_pid")

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

    model_params = set(analysis_obj.cost_details.constant_values) - set(
        analysis_obj.analysis_details.excluded_parameters
    )

    optimization_data = analysis_obj.query_optimization_data(
        excluded_parameters=analysis_obj.analysis_details.excluded_parameters
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
