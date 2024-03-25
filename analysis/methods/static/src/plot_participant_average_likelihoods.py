import logging
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pingouin as pg
import seaborn as sns
from costometer.utils import (
    get_correlation_text,
    get_static_palette,
    get_wilcoxon_text,
    standard_parse_args,
)


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
    irl_path = Path(__file__).resolve().parents[4]
    analysis_obj, inputs, subdirectory = standard_parse_args(
        description=sys.modules[__name__].__doc__,
        irl_path=irl_path,
        filename=Path(__file__).stem,
    )

    trial_by_trial_df = analysis_obj.get_trial_by_trial_likelihoods()

    # look only at relevant block
    mouselab_data = analysis_obj.dfs["mouselab-mdp"]
    relevant_trials = mouselab_data[
        mouselab_data["block"].isin(analysis_obj.analysis_details.blocks)
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

    logging.info("==========")
    logging.info(
        "Number of participants, for top three models, where that model explains "
        "them the best (via meta-level action likelihoods)"
    )
    logging.info("----------")
    logging.info(
        Counter(
            participant_df.loc[participant_df.groupby(["pid"]).idxmax()["avg"]][
                "Model Name"
            ]
        )
    )

    logging.info("==========")
    for model in participant_df["Model Name"].unique():
        if model != best_model:
            logging.info("----------")
            logging.info(
                "Difference between meta-level action"
                " likelihoods for best model and %s",
                model,
            )
            logging.info("----------")
            wilcoxon_object = pg.wilcoxon(
                participant_df[participant_df["Model Name"] == model]["avg"],
                participant_df[participant_df["Model Name"] == best_model]["avg"],
                alternative="two-sided",
            )
            logging.info(
                "%s; $M_{%s} = %.3f$; $M_{%s} = %.3f$",
                get_wilcoxon_text(wilcoxon_object),
                best_model.replace("$", ""),
                np.median(
                    participant_df[participant_df["Model Name"] == best_model]["avg"]
                ),
                model.replace("$", ""),
                np.median(participant_df[participant_df["Model Name"] == model]["avg"]),
            )

    logging.info("==========")
    logging.info("Meta-level action table")
    logging.info("----------")
    for row_idx, row in (
        participant_df.groupby(["Model Name"])
        .describe()["avg"]
        .sort_values(by="mean", ascending=False)
        .iterrows()
    ):
        logging.info(
            "%s & %.3f & %.3f \\ ",  # noqa : W605
            row_idx,
            row["mean"],
            row["std"],
        )

    logging.info("==========")
    logging.info("Correlation between episode and meta-level action log likelihood")
    logging.info("----------")
    correlation = pg.corr(
        trial_by_trial_df[(trial_by_trial_df["Model Name"] == best_model)]
        .groupby("i_episode", as_index=False)
        .mean()["avg"],
        trial_by_trial_df[(trial_by_trial_df["Model Name"] == best_model)]
        .groupby("i_episode", as_index=False)
        .mean()["i_episode"],
    )
    logging.info(get_correlation_text(correlation))
