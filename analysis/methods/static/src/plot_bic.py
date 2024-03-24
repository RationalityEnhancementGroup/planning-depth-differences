import logging
import sys
from pathlib import Path

import dill as pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from costometer.utils import get_static_palette, standard_parse_args
from statsmodels.tools.eval_measures import bic


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
    irl_path = Path(__file__).resolve().parents[4]
    analysis_obj, inputs, subdirectory = standard_parse_args(
        description=sys.modules[__name__].__doc__,
        irl_path=irl_path,
        filename=Path(__file__).stem,
    )

    optimization_data = analysis_obj.query_optimization_data()

    excluded_set = set(analysis_obj.analysis_details.excluded_parameters)

    optimization_data = optimization_data[
        optimization_data.apply(
            lambda row: excluded_set.issubset(row["model"])
            or (row["Model Name"] == "Null"),
            axis=1,
        )
    ]

    bic_df = (
        optimization_data.groupby(["Model Name", "Number Parameters"])
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

    logging.info(bic_df.sort_values(by="bic").round(5))

    if (
        hasattr(analysis_obj, "simulated_bic")
        and irl_path.joinpath(
            "analysis/methods/static/data/Simulated_BIC.pickle"
        ).is_file()
    ):
        with open(
            irl_path.joinpath("analysis/methods/static/data/Simulated_BIC.pickle"), "rb"
        ) as f:
            simulated_means = pickle.load(f)["SimulatedParticipant"]
    else:
        simulated_means = None

    bic_plot(
        bic_df,
        subdirectory,
        experiment_name=analysis_obj.analysis_details.palette_name
        if analysis_obj.analysis_details.palette_name
        else inputs.experiment_name,
        bic_field="bic",
    )
    if simulated_means:
        plt.axvline(x=np.mean(simulated_means))
    title_extras = f" ({', '.join(analysis_obj.analysis_details.title_extras)})"
    plt.title(
        f"Bayesian Information Criterion"
        f"{title_extras if analysis_obj.analysis_details.title_extras else ''}"
    )
    plt.savefig(
        subdirectory.joinpath(f"figs/{inputs.experiment_name}_bic.png"),
        bbox_inches="tight",
    )

    # Bayes Factor approximation
    logging.info("Log Bayes factor approximation, difference between top two models")
    logging.info(
        (bic_df["bic"].sort_values().iloc[1] - bic_df["bic"].sort_values().iloc[0]) / 2
    )

    logging.info(
        "Log Bayes factor approximation, difference between top and third models"
    )
    logging.info(
        (bic_df["bic"].sort_values().iloc[2] - bic_df["bic"].sort_values().iloc[0]) / 2
    )
