from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pingouin as pg
import seaborn as sns
from costometer.utils import (
    AnalysisObject,
    get_correlation_text,
    get_static_palette,
    get_wilcoxon_text,
    set_font_sizes,
)

set_font_sizes()

###################################################
# This section contains my plotting functions
###################################################


def plot_participant_average_likelihoods(
    optimization_data,
    likelihood_field,
    subdirectory,
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
    plt.ylabel("Average action likelihood (log scale, last twenty trials)")
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
    subdirectory,
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
    subdirectory = irl_path.joinpath(f"analysis/{inputs.experiment_subdirectory}")

    analysis_obj = AnalysisObject(
        inputs.experiment_name,
        irl_path=irl_path,
        experiment_subdirectory=inputs.experiment_subdirectory,
    )

    trial_by_trial_df = analysis_obj.get_trial_by_trial_likelihoods()

    # look only at relevant block
    mouselab_data = analysis_obj.dfs["mouselab-mdp"]
    relevant_trials = mouselab_data[
        mouselab_data["block"].isin(analysis_obj.block.split(","))
    ]["trial_index"].unique()
    trial_by_trial_df = trial_by_trial_df[
        trial_by_trial_df["i_episode"].isin(relevant_trials)
    ]

    best_model = trial_by_trial_df[trial_by_trial_df["best_model"] == 1][
        "Model Name"
    ].unique()[0]

    plot_participant_average_likelihoods(
        trial_by_trial_df,
        "avg",
        subdirectory,
        experiment_name=inputs.experiment_name,
        dodge=0.25,
    )
    plt.savefig(
        subdirectory.joinpath(
            f"figs/{inputs.experiment_name}_participant_lik_twenty.png"
        ),
        bbox_inches="tight",
    )

    plot_trial_by_trial_logliks(
        trial_by_trial_df,
        "avg",
        subdirectory,
        experiment_name=inputs.experiment_name,
        agg_func=np.mean,
        dodge=0.25,
    )
    plt.savefig(subdirectory.joinpath(f"figs/{inputs.experiment_name}_average_ll.png"))

    plot_trial_by_trial_logliks(
        trial_by_trial_df,
        "avg",
        subdirectory,
        experiment_name=inputs.experiment_name,
        agg_func=np.sum,
        dodge=0.25,
    )
    plt.savefig(subdirectory.joinpath(f"figs/{inputs.experiment_name}_total_ll.png"))

    participant_df = (
        trial_by_trial_df.groupby(["Model Name", "pid"]).mean().reset_index()
    )

    max_avg = participant_df.loc[participant_df.groupby(["pid"]).idxmax()["avg"]][
        ["pid", "Model Name", "avg"]
    ].values.tolist()

    unique_models_per_pid = (
        participant_df[
            participant_df.apply(
                lambda row: [row["pid"], row["Model Name"], row["avg"]] in max_avg,
                axis=1,
            )
        ]
        .groupby(["pid"])
        .nunique()["Model Name"]
    )
    assert unique_models_per_pid.unique() == [1]

    print("==========")
    print(
        "Number of participants, for top three models, where that model explains "
        "them the best (via meta-level action likelihoods)"
    )
    print("----------")
    print(
        Counter(
            participant_df.loc[participant_df.groupby(["pid"]).idxmax()["avg"]][
                "Model Name"
            ]
        )
    )

    print("==========")
    for model in participant_df["Model Name"].unique():
        if model != best_model:
            print("----------")
            print(
                f"Difference between meta-level action likelihoods "
                f"for best model and {model}"
            )
            print("----------")
            wilcoxon_object = pg.wilcoxon(
                participant_df[participant_df["Model Name"] == model]["avg"],
                participant_df[participant_df["Model Name"] == best_model]["avg"],
                alternative="two-sided",
            )
            print(
                f"M_{{\\text{{{model}}}}} = {np.median(participant_df[participant_df['Model Name'] == model]['avg']):.2f}\n"  # noqa: E501
                f"M_{{\\text{{{best_model}}}}} ="
                f"{np.median(participant_df[participant_df['Model Name'] == best_model]['avg']):.2f}"  # noqa: E501
            )
            print(get_wilcoxon_text(wilcoxon_object))

    effort_costs = (
        participant_df.groupby(["Model Name"]).describe()["avg"].reset_index()
    )
    effort_costs["mean"] = effort_costs["mean"].apply(lambda entry: f"{entry:.3f}")
    effort_costs["std"] = effort_costs["std"].apply(lambda entry: f"{entry:.3f}")

    print("==========")
    print("Meta-level action table")
    print("----------")
    for row_idx, row in effort_costs.sort_values(by="mean", ascending=False).iterrows():
        print(f"{row['Model Name']} & {row['mean']} & {row['std']} \\\ ")  # noqa

    print("==========")
    print("Correlation between episode and meta-level action log likelihood")
    print("----------")
    correlation = pg.corr(
        trial_by_trial_df[(trial_by_trial_df["Model Name"] == best_model)]["avg"],
        trial_by_trial_df[(trial_by_trial_df["Model Name"] == best_model)]["i_episode"],
    )
    print(get_correlation_text(correlation))
