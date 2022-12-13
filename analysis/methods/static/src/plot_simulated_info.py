"""
Plots information about the number of nodes clicked by the optimal policy
using the Q values under different cost parameters
"""
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from costometer.utils import set_font_sizes

set_font_sizes()


def plot_heat_map_for_simulated(sum_df, field, rew1, rew2, name1, name2):
    plt.figure(figsize=(16, 12))
    heat_map_data = sum_df.pivot(index=rew1, columns=rew2, values=field)
    sns.heatmap(data=heat_map_data, annot=True, fmt=".2f")
    plt.ylabel(name2)
    plt.xlabel(name1)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--cost", dest="cost_function", default="linear_depth")
    parser.add_argument(
        "-e",
        "--experiment-setting",
        dest="experiment_setting",
        default="high_increasing",
    )
    inputs = parser.parse_args()

    irl_path = Path(__file__).resolve().parents[4]
    static_directory = Path(__file__).resolve().parents[1]

    optimal_df = pd.read_csv(
        irl_path.joinpath(f"cluster/data/OptimalQ/OptimalQ_{inputs.cost_function}.csv")
    )

    yaml_file = irl_path.joinpath(
        f"data/inputs/yamls/cost_functions/{inputs.cost_function}.yaml"
    )
    with open(str(yaml_file), "r") as stream:
        cost_details = yaml.safe_load(stream)

    sim_cost_parameters = [
        f"sim_{param}" for param in cost_details["constant_values"].keys()
    ]
    cost_parameter_names = cost_details["cost_parameter_names"]

    yaml_file = irl_path.joinpath(
        f"data/inputs/yamls/experiment_settings/{inputs.experiment_setting}.yaml"
    )
    with open(str(yaml_file), "r") as stream:
        experiment_setting_details = yaml.safe_load(stream)

    for classification, nodes in experiment_setting_details[
        "node_classification"
    ].items():
        optimal_df[classification] = optimal_df["actions"].apply(
            lambda action: action in nodes
        )

    sum_clicks = (
        optimal_df.groupby(["pid", "i_episode", *sim_cost_parameters])
        .sum()
        .reset_index()
        .groupby(sim_cost_parameters)
        .mean()
        .reset_index()
    )

    for curr_field in list(experiment_setting_details["node_classification"].keys()):
        if len(sim_cost_parameters) == 2:
            plot_heat_map_for_simulated(
                sum_clicks,
                curr_field,
                *sim_cost_parameters,
                *cost_parameter_names,
            )
            plt.title(f"{curr_field.title()}")
            plt.savefig(
                static_directory.joinpath(f"figs/optimal_{curr_field}.png"),
                bbox_inches="tight",
            )
        else:
            held_cost_parameter = sim_cost_parameters[2]
            held_cost_parameter_name = cost_parameter_names[2]
            for held_cost_parameter_value in sum_clicks[held_cost_parameter].unique():
                curr_sum_df = sum_clicks[
                    sum_clicks[held_cost_parameter] == held_cost_parameter_value
                ].reset_item()
                plot_heat_map_for_simulated(
                    curr_sum_df,
                    curr_field,
                    *sim_cost_parameters,
                    *cost_parameter_names,
                )
                plt.title(
                    f"{curr_field.title()}, {held_cost_parameter_name}"
                    f"={held_cost_parameter_value}"
                )
                plt.savefig(
                    static_directory.joinpath(
                        f"figs/optimal_{curr_field}_{held_cost_parameter_name}"
                        f"_{held_cost_parameter_value}.png"
                    ),
                    bbox_inches="tight",
                )
