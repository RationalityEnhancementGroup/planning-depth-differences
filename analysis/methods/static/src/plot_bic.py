from argparse import ArgumentParser
from pathlib import Path

import dill as pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from costometer.utils import AnalysisObject, get_static_palette, set_font_sizes
from statsmodels.tools.eval_measures import bic

set_font_sizes()

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
        data=optimization_data.sort_values(by="bic").iloc[:16],
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

    print(bic_df.sort_values(by="bic").round(5))

    if irl_path.joinpath("analysis/methods/" "static/data/OptimalBIC.pickle").is_file():
        with open(
            irl_path.joinpath("analysis/methods/" "static/data/OptimalBIC.pickle"), "rb"
        ) as f:
            simulated_means = pickle.load(f)["intended"]
    else:
        simulated_means = None

    bic_plot(
        bic_df,
        subdirectory,
        experiment_name=inputs.experiment_name,
        bic_field="bic",
    )
    if simulated_means:
        plt.axvline(x=np.mean(simulated_means))
    title_extras = f" ({', '.join(analysis_obj.title_extras)})"
    plt.title(
        f"Bayesian Information Criterion"
        f"{title_extras if analysis_obj.title_extras else ''}"
    )
    plt.savefig(
        subdirectory.joinpath(f"figs/{inputs.experiment_name}_bic.png"),
        bbox_inches="tight",
    )

    # Bayes Factor approximation
    print("Log Bayes factor approximation, difference between top two models")
    print(
        (bic_df["bic"].sort_values().iloc[1] - bic_df["bic"].sort_values().iloc[0]) / 2
    )
