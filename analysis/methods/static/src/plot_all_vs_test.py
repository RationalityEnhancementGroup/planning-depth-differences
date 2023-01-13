from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pingouin as pg
import seaborn as sns
from costometer.utils import (
    AnalysisObject,
    get_static_palette,
    get_trajectories_from_participant_data,
    get_wilcoxon_text,
    set_font_sizes,
    traces_to_df,
)

set_font_sizes()

###################################################
# This section contains my plotting functions
###################################################


def plot_participant_average_likelihoods(
    optimization_data,
    subdirectory,
    likelihood_field,
    experiment_name,
    palette=None,
    dodge=False,
):
    if palette is None:
        palette = get_static_palette(subdirectory, experiment_name)
    plt.figure(figsize=(16, 8), dpi=80)
    ax = sns.pointplot(
        y=likelihood_field,
        x="pid",
        hue="Model Name",
        linestyles="none",
        palette=palette,
        dodge=dodge,
        data=optimization_data,
        order=optimization_data.groupby("pid")[likelihood_field]
        .agg("mean")
        .sort_values()
        .index,
    )

    avg_trial_lik = optimization_data.groupby(["Model Name"])[likelihood_field].mean()
    for model, color in palette.items():
        if model in avg_trial_lik:
            plt.axhline(y=avg_trial_lik[model], color=color)

    plt.tight_layout()

    plt.xticks([], [])
    plt.xlabel("Participant")
    plt.ylabel("Average action likelihood (log scale, last ten trials)")
    ax.legend(loc="best")
    ax.set_yscale("log")

    # https://stackoverflow.com/a/27512450
    label_order = (
        optimization_data.groupby(["Model Name"]).sum()[likelihood_field].to_dict()
    )

    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(
        *sorted(zip(labels, handles), key=lambda t: label_order[t[0]])
    )
    ax.legend(handles[::-1], labels[::-1])


def plot_trial_by_trial_logliks(
    optimization_data,
    likelihood_field,
    experiment_name,
    agg_func=np.mean,
    palette=None,
    dodge=False,
):
    if palette is None:
        palette = get_static_palette(subdirectory, experiment_name)
    plt.figure(figsize=(11.7, 8.27))
    ax = sns.pointplot(
        y=likelihood_field,
        x="i_episode",
        hue="Model Name",
        dodge=dodge,
        palette=palette,
        estimator=agg_func,
        data=optimization_data,
    )
    ax.set_yscale("log")
    plt.xlabel("Trial")
    plt.ylabel("Average action likelihood (log scale)")
    if agg_func == np.sum:
        plt.title("Total action likelihood per trial")
    elif agg_func == np.mean:
        plt.title("Average action likelihood per trial")
    plt.xticks([], [])

    # https://stackoverflow.com/a/27512450
    label_order = (
        optimization_data.groupby(["Model Name"]).sum()[likelihood_field].to_dict()
    )

    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(
        *sorted(zip(labels, handles), key=lambda t: label_order[t[0]])
    )
    ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-e", "--exp1", dest="experiment_name1", default="TrialByTrial")
    parser.add_argument(
        "-f", "--exp2", dest="experiment_name2", default="TrialByTrialAll"
    )
    parser.add_argument(
        "-s",
        "--subdirectory",
        dest="experiment_subdirectory",
        metavar="experiment_subdirectory",
    )

    inputs = parser.parse_args()

    irl_path = Path(__file__).resolve().parents[4]
    subdirectory = irl_path.joinpath(f"analysis/{inputs.experiment_subdirectory}/data")

    analysis_obj1 = AnalysisObject(
        inputs.experiment_name1,
        irl_path=irl_path,
        experiment_subdirectory=inputs.experiment_subdirectory,
    )
    analysis_obj2 = AnalysisObject(
        inputs.experiment_name2,
        irl_path=irl_path,
        experiment_subdirectory=inputs.experiment_subdirectory,
    )

    trace_df1 = traces_to_df(
        get_trajectories_from_participant_data(analysis_obj1.mouselab_trials)
    )
    num_actions1 = (
        trace_df1.groupby(["pid", "i_episode"])
        .count()["actions"]
        .reset_index()
        .rename(columns={"actions": "num_actions"})
    )

    trace_df2 = traces_to_df(
        get_trajectories_from_participant_data(analysis_obj2.mouselab_trials)
    )
    num_actions2 = (
        trace_df2.groupby(["pid", "i_episode"])
        .count()["actions"]
        .reset_index()
        .rename(columns={"actions": "num_actions"})
    )

    trial_by_trial_df1 = analysis_obj1.get_trial_by_trial_likelihoods()
    trial_by_trial_df1 = trial_by_trial_df1.merge(num_actions1, on=["pid", "i_episode"])

    trial_by_trial_df2 = analysis_obj2.get_trial_by_trial_likelihoods()
    trial_by_trial_df2 = trial_by_trial_df2.merge(num_actions2, on=["pid", "i_episode"])

    trial_by_trial_df1["avg"] = trial_by_trial_df1.apply(
        lambda row: np.exp(row["likelihood"]) / row["num_actions"], axis=1
    )

    relevant_trials = analysis_obj1.mouselab_trials[
        analysis_obj1.mouselab_trials["block"].isin(analysis_obj1.block)
    ]["trial_index"]

    trial_by_trial_df1 = trial_by_trial_df1[
        trial_by_trial_df1["i_episode"].isin(relevant_trials)
    ]

    trial_by_trial_df2["avg"] = trial_by_trial_df2.apply(
        lambda row: np.exp(row["likelihood"]) / row["num_actions"], axis=1
    )

    trial_by_trial_df1 = trial_by_trial_df1[
        trial_by_trial_df1["Model Name"] == "'Distance, Effort, Depth and Forward Search Bonus'"
    ]
    trial_by_trial_df2 = trial_by_trial_df2[
        trial_by_trial_df2["Model Name"] == "'Distance, Effort, Depth and Forward Search Bonus'"
    ]

    for model in trial_by_trial_df1["Model Name"].unique():
        print(f"Mean and standard deviation for {model} last 20 trials"),
        model_subset = trial_by_trial_df1[(trial_by_trial_df1["Model Name"] == model)]
        print(
            f"$M: {model_subset['avg'].mean():.2f}, "
            f"SD: {model_subset['avg'].std():.2f}$"
        )

    for model in trial_by_trial_df2["Model Name"].unique():
        print(f"Mean and standard deviation for {model} all trials")
        model_subset = trial_by_trial_df2[(trial_by_trial_df2["Model Name"] == model)]
        print(
            f"$M: {model_subset['avg'].mean():.2f}, "
            f"SD: {model_subset['avg'].std():.2f}$"
        )

    plot_participant_average_likelihoods(
        trial_by_trial_df2[trial_by_trial_df2["i_episode"].isin(relevant_trials)],
        subdirectory,
        "avg",
        experiment_name=inputs.experiment_name,
        dodge=0.25,
    )
    plt.savefig(
        subdirectory.joinpath(
            f"figs/{inputs.experiment_name2}_participant_lik_ten.png"
        ),
        bbox_inches="tight",
    )

    comparison = pg.wilcoxon(
        trial_by_trial_df1.groupby(["pid"], as_index=False)
        .mean()
        .sort_values(["pid", "i_episode"])["avg"],
        trial_by_trial_df2.groupby(["pid"], as_index=False)
        .mean()
        .sort_values(["pid", "i_episode"])["avg"],
    )

    print(get_wilcoxon_text(comparison))
