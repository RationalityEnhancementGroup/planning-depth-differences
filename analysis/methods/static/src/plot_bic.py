from argparse import ArgumentParser
from pathlib import Path

import dill as pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from costometer.utils import AnalysisObject, get_static_palette, set_font_sizes

set_font_sizes()

###################################################
# This section contains my plotting functions
###################################################


def bic_plot(optimization_data, static_directory, bic_field="bic", palette=None):
    if palette is None:
        palette = get_static_palette(static_directory)
    plt.figure(figsize=(12, 8), dpi=80)
    sum_bic = optimization_data.groupby(["Model Name"])[bic_field].sum()
    order = sum_bic.sort_values().index
    sns.barplot(
        y="Model Name",
        x=bic_field,
        estimator=np.sum,
        ci=None,
        data=optimization_data,
        order=order,
        palette=palette,
    )

    plt.xlabel(None)
    plt.ylabel(None)

    plt.gca().set_xlim((0.75 * min(sum_bic), plt.gca().get_xlim()[1]))


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
    inputs = parser.parse_args()

    static_directory = Path(__file__).resolve().parents[1]
    data_path = Path(__file__).resolve().parents[1]
    irl_path = Path(__file__).resolve().parents[4]

    analysis_obj = AnalysisObject(inputs.experiment_name, irl_path=irl_path)

    optimization_data = analysis_obj.query_optimization_data()

    # sum over pids
    bic_df = (
        optimization_data.groupby(["Model Name", "cost_function"])
        .sum()["bic"]
        .reset_index()
    )

    with open(data_path.joinpath("data/OptimalBIC.pickle"), "rb") as f:
        simulated_means = pickle.load(f)["intended"]

    bic_plot(bic_df, static_directory, bic_field="bic")
    plt.axvline(x=np.mean(simulated_means))
    title_extras = f" ({', '.join(analysis_obj.title_extras)})"
    plt.title(
        f"Bayesian Information Criterion"
        f"{title_extras if analysis_obj.title_extras else ''}"
    )
    plt.savefig(
        static_directory.joinpath(f"figs/{inputs.experiment_name}_bic.png"),
        bbox_inches="tight",
    )

    # Bayes Factor approximation
    print("Log Bayes factor approximation, difference between top two models")
    print(bic_df["bic"].sort_values().iloc[1] - bic_df["bic"].sort_values().iloc[0])
