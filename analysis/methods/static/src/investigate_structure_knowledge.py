from argparse import ArgumentParser
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pingouin as pg
import seaborn as sns
from costometer.utils import (
    AnalysisObject,
    get_kruskal_wallis_text,
    get_mann_whitney_text,
    get_static_palette,
    set_font_sizes,
)

set_font_sizes()

###################################################
# This section contains my plotting function(s)
###################################################


def plot_score_average_likelihoods(
    optimization_data_trials,
    likelihood_field,
    subdirectory,
    experiment_name,
    palette=None,
):
    if palette is None:
        palette = get_static_palette(subdirectory, experiment_name)
    plt.figure(figsize=(12, 8), dpi=80)
    ax = sns.pointplot(
        y=likelihood_field,
        x="score",
        hue="Model Name",
        linestyles="none",
        palette=palette,
        data=optimization_data_trials,
    )

    avg_trial_lik = optimization_data_trials.groupby(["Model Name"])[
        likelihood_field
    ].mean()
    for model, color in palette.items():
        if model in avg_trial_lik:
            plt.axhline(y=avg_trial_lik[model], color=color)

    plt.tight_layout()

    plt.xlabel("Post-test Score")
    pid_counts = (
        optimization_data_trials.drop_duplicates("trace_pid")
        .groupby(["score"])
        .count()["trace_pid"]
        .to_dict()
    )
    new_xtick_labels = []
    for xtick_label in ax.get_xticklabels():
        curr_label = xtick_label._text
        new_xtick_labels.append(f"{curr_label} (n={pid_counts[float(curr_label)]})")
    ax.set_xticklabels(new_xtick_labels)
    ax.legend(loc="best")


if __name__ == "__main__":
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
    trial_by_trial_df = analysis_obj.get_trial_by_trial_likelihoods()

    best_model = trial_by_trial_df[trial_by_trial_df["best_model"] == 1][
        "Model Name"
    ].unique()[0]

    mouselab_data = analysis_obj.dfs["mouselab-mdp"]
    relevant_trials = mouselab_data[
        mouselab_data["block"].isin(analysis_obj.block.split(","))
    ]["trial_index"].unique()

    model_params = list(analysis_obj.cost_details["constant_values"])

    melted_df = analysis_obj.dfs["quiz-and-demo"].melt(
        id_vars=["pid", "run"], value_vars=analysis_obj.post_quizzes, value_name="score"
    )
    melted_df = melted_df.dropna(subset=["score"])
    scores = melted_df.groupby(["pid"]).sum()["score"].reset_index()
    full_score_pids = scores[
        scores["score"] == len(analysis_obj.post_quizzes)
    ].pid.unique()
    print(f"Number of participants with full score: {len(full_score_pids)}")

    optimization_data = optimization_data.merge(
        trial_by_trial_df[trial_by_trial_df["i_episode"].isin(relevant_trials)]
        .groupby(["pid", "Model Name"])
        .mean()["avg"]
        .reset_index(),
        left_on=["trace_pid", "Model Name"],
        right_on=["pid", "Model Name"],
    )
    optimization_data = optimization_data.merge(scores, on=["pid"])

    plot_score_average_likelihoods(
        optimization_data,
        "avg",
        subdirectory,
        experiment_name=inputs.experiment_name,
    )
    plt.ylabel("Average planning operation likelihood")
    plt.savefig(
        subdirectory.joinpath(f"figs/{inputs.experiment_name}_score_lik.png"),
        bbox_inches="tight",
    )

    optimization_data = optimization_data[optimization_data["Model Name"] == best_model]

    for obs_var in ["avg"] + model_params:
        print("==========")
        print(f"Comparisons for {obs_var}")
        omnibus = pg.kruskal(data=optimization_data, dv=obs_var, between="score")
        print("----------")
        print("Omnibus test")
        print(get_kruskal_wallis_text(omnibus))

        print("----------")
        print(f"Pair-wise comparisons: {obs_var}")
        print("----------")
        pairs = combinations(optimization_data["score"].unique(), 2)

        for pair in pairs:
            score1, score2 = pair
            print("----------")
            print(f"Pair-wise comparison: {obs_var} {score1}, {score2}")
            print("----------")

            pair_comparison = pg.mwu(
                optimization_data[optimization_data["score"] == score1][obs_var],
                optimization_data[optimization_data["score"] == score2][obs_var],
            )

            print(get_mann_whitney_text(pair_comparison))

            mean1 = np.mean(
                optimization_data[optimization_data["score"] == score1][obs_var]
            )
            mean2 = np.mean(
                optimization_data[optimization_data["score"] == score2][obs_var]
            )
            print(
                f"$M_{{\\text{{{score1:.0f}}}}}={mean1:.2f}$ "
                f"vs $M_{{\\text{{{score2:.0f}}}}}={mean2:.2f}$"
            )
