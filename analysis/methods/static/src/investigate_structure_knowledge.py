from argparse import ArgumentParser
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
    get_trajectories_from_participant_data,
    set_font_sizes,
    traces_to_df,
)

set_font_sizes()

###################################################
# This section contains my plotting function(s)
###################################################


def plot_score_average_likelihoods(
    optimization_data_trials, likelihood_field, static_directory, palette=None
):
    if palette is None:
        palette = get_static_palette(static_directory)
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
    inputs = parser.parse_args()

    static_directory = Path(__file__).resolve().parents[1]
    irl_path = Path(__file__).resolve().parents[4]

    analysis_obj = AnalysisObject(inputs.experiment_name, irl_path=irl_path)

    trace_df = traces_to_df(
        get_trajectories_from_participant_data(analysis_obj.mouselab_trials)
    )
    num_actions = (
        trace_df.groupby(["pid", "i_episode"])
        .count()["actions"]
        .reset_index()
        .rename(columns={"actions": "num_actions"})
    )

    trial_by_trial_df = analysis_obj.get_trial_by_trial_likelihoods()
    trial_by_trial_df = trial_by_trial_df.merge(num_actions, on=["pid", "i_episode"])

    trial_by_trial_df["avg"] = trial_by_trial_df.apply(
        lambda row: np.exp(row["likelihood"] / row["num_actions"]), axis=1
    )

    relevant_trials = analysis_obj.mouselab_trials[
        analysis_obj.mouselab_trials["block"].isin(analysis_obj.block)
    ]["trial_index"]

    melted_df = analysis_obj.quest.melt(
        id_vars=["pid", "run"], value_vars=analysis_obj.post_quizzes, value_name="score"
    )
    melted_df = melted_df.dropna(subset=["score"])
    scores = melted_df.groupby(["pid"]).sum()["score"].reset_index()
    full_score_pids = scores[
        scores["score"] == len(analysis_obj.post_quizzes)
    ].pid.unique()

    optimization_data = analysis_obj.query_optimization_data()
    optimization_data = optimization_data.merge(
        trial_by_trial_df[trial_by_trial_df["i_episode"].isin(relevant_trials)]
        .groupby(["pid", "Model Name"])
        .mean()["avg"]
        .reset_index(),
        left_on=["trace_pid", "Model Name"],
        right_on=["pid", "Model Name"],
    )
    optimization_data = optimization_data.merge(scores, on=["pid"])

    plot_score_average_likelihoods(optimization_data, "avg", static_directory)
    plt.ylabel("Average planning operation likelihood")
    plt.savefig(
        static_directory.joinpath(f"figs/{inputs.experiment_name}_score_lik.png"),
        bbox_inches="tight",
    )

    optimization_data = optimization_data[
        optimization_data["Model Name"] == "Effort Cost and Planning Depth"
    ]

    for obs_var in ["avg", "static_cost_weight", "depth_cost_weight"]:
        print("==========")
        print(f"Comparisons for {obs_var}")
        omnibus = pg.kruskal(data=optimization_data, dv=obs_var, between="score")
        print("----------")
        print("Omnibus test")
        print(get_kruskal_wallis_text(omnibus))

        print("----------")
        comparison_perfect = pg.mwu(
            optimization_data[optimization_data["score"] == 4][obs_var],
            optimization_data[optimization_data["score"] < 4][obs_var],
        )
        print("Comparison for perfect post-test scorers vs everyone else")
        print("----------")
        print(get_mann_whitney_text(comparison_perfect))
        mean_perfect = np.mean(
            optimization_data[optimization_data["score"] == 4][obs_var]
        )
        mean_other = np.mean(optimization_data[optimization_data["score"] < 4][obs_var])
        print(
            f"$M_{{\\text{{perfect}}}}={mean_perfect:.2f}$ vs "
            f"$M_{{\\text{{other}}}}={mean_other:.2f}$"
        )
