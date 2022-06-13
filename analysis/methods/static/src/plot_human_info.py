from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
import yaml
from costometer.utils import (
    AnalysisObject,
    get_correlation_text,
    get_trajectories_from_participant_data,
    set_font_sizes,
    traces_to_df,
)

set_font_sizes()


def plot_heat_map_for_human(sum_df, field):
    plt.figure(figsize=(16, 12))
    heat_map_data = sum_df.pivot(
        index="static_cost_weight", columns="depth_cost_weight", values=field
    )
    sns.heatmap(data=heat_map_data, annot=True, fmt=".2f")
    plt.ylabel("Planning Depth Cost")
    plt.xlabel("Effort Cost")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp",
        dest="experiment_name",
        metavar="experiment_name",
    )

    inputs = parser.parse_args()

    irl_path = Path(__file__).resolve().parents[4]
    static_directory = Path(__file__).resolve().parents[1]

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

    assert (
        len(
            np.unique(
                [
                    session_details["experiment_setting"]
                    for session_details in analysis_obj.session_details.values()
                ]
            )
        )
        == 1
    )
    experiment_setting = np.unique(
        [
            session_details["experiment_setting"]
            for session_details in analysis_obj.session_details.values()
        ]
    )[0]

    with open(
        irl_path.joinpath(
            f"data/inputs/yamls/experiment_settings/{experiment_setting}.yaml"
        ),
        "rb",
    ) as f:
        experiment_setting_details = yaml.safe_load(f)

    # add node classification columns
    for classification, nodes in experiment_setting_details[
        "node_classification"
    ].items():
        trace_df[classification] = trace_df["actions"].apply(
            lambda action: action in nodes
        )

    optimization_data = analysis_obj.query_optimization_data()
    optimization_data = optimization_data[
        optimization_data["Model Name"] == "Effort Cost and Planning Depth"
    ]

    trace_df = trace_df.merge(optimization_data, left_on="pid", right_on="trace_pid")

    sum_clicks = (
        trace_df.groupby(
            ["pid", "i_episode", "static_cost_weight", "depth_cost_weight", "trial_id"]
        )
        .sum()
        .reset_index()
        .groupby(["static_cost_weight", "depth_cost_weight"])
        .mean()
        .reset_index()
    )

    for curr_field in experiment_setting_details["node_classification"].keys():
        plot_heat_map_for_human(sum_clicks, curr_field)
        plt.savefig(
            static_directory.joinpath(f"figs/ppc_{curr_field}.png"),
            bbox_inches="tight",
        )

    sum_over_pids = (
        trace_df.groupby(
            [
                "pid",
                "i_episode",
                "static_cost_weight",
                "depth_cost_weight",
                "temp",
            ]
        )
        .sum()
        .reset_index()
        .groupby(
            [
                "pid",
                "static_cost_weight",
                "depth_cost_weight",
                "temp",
            ]
        )
        .mean()
        .reset_index()
    )

    static_directory.joinpath("processed/human").mkdir(parents=True, exist_ok=True)
    sum_over_pids.to_csv(
        static_directory.joinpath(f"processed/human/{inputs.experiment_name}_bias.csv")
    )

    cost_function = optimization_data["cost_function"].unique()[0]
    optimal_df = pd.read_csv(
        irl_path.joinpath(f"cluster/data/OptimalQ/OptimalQ_{cost_function}.csv")
    )

    # add node classification columns
    for classification, nodes in experiment_setting_details[
        "node_classification"
    ].items():
        optimal_df[classification] = optimal_df["actions"].apply(
            lambda action: action in nodes
        )

    mean_over_cost = (
        optimal_df.groupby(
            ["pid", "i_episode", "sim_static_cost_weight", "sim_depth_cost_weight"]
        )
        .sum()
        .reset_index()
        .groupby(["sim_static_cost_weight", "sim_depth_cost_weight"])
        .mean()
        .reset_index()
    )
    sum_over_pids = sum_over_pids.merge(
        mean_over_cost,
        left_on=["static_cost_weight", "depth_cost_weight"],
        right_on=["sim_static_cost_weight", "sim_depth_cost_weight"],
        suffixes=("", "_optimal"),
    )

    for classification, nodes in experiment_setting_details[
        "node_classification"
    ].items():
        print(
            f"Correlation of metric '{classification}' between "
            f"simulated and real data, per participant"
        )
        correlation_obj = pg.corr(
            sum_over_pids[f"{classification}"],
            sum_over_pids[f"{classification}_optimal"],
        )
        print(get_correlation_text(correlation_obj))

    sum_over_params = (
        sum_over_pids.groupby(
            analysis_obj.cost_details["linear_depth"]["cost_parameter_args"]
        )
        .mean()
        .reset_index()
    )
    for classification, nodes in experiment_setting_details[
        "node_classification"
    ].items():
        print(
            f"Correlation of metric '{classification}' between "
            f"simulated and real data, per cost setting"
        )
        correlation_obj = pg.corr(
            sum_over_params[f"{classification}"],
            sum_over_params[f"{classification}_optimal"],
        )
        print(get_correlation_text(correlation_obj))
