"""
Plots information about the number of nodes clicked by the optimal policy
using the Q values under different cost parameters
"""
import itertools
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from costometer.utils import AnalysisObject, set_font_sizes

set_font_sizes()


def plot_heat_map_for_simulated(sum_df, field, rew1, rew2, name1, name2):
    plt.figure(figsize=(16, 12))
    heat_map_data = sum_df.pivot(index=rew1, columns=rew2, values=field)
    sns.heatmap(data=heat_map_data, annot=True, fmt=".2f")
    plt.ylabel(name2)
    plt.xlabel(name1)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp",
        dest="experiment_name",
        default="SoftmaxRecovery",
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

    mouselab_data = analysis_obj.dfs["mouselab-mdp"]

    sim_cost_parameters = [
        f"sim_{param}" for param in analysis_obj.cost_details["constant_values"].keys()
    ]

    yaml_file = irl_path.joinpath(
        f"data/inputs/yamls/experiment_settings/{analysis_obj.experiment_setting}.yaml"
    )
    with open(str(yaml_file), "r") as stream:
        experiment_setting_details = yaml.safe_load(stream)

    for classification, nodes in experiment_setting_details[
        "node_classification"
    ].items():
        mouselab_data[classification] = mouselab_data["actions"].apply(
            lambda action: action in nodes
        )

    sum_clicks = (
        mouselab_data.groupby(["pid", "i_episode", *sim_cost_parameters])
        .sum()
        .reset_index()
        .groupby(sim_cost_parameters)
        .mean()
        .reset_index()
    )

    sim_latex_mapping = {
        f"sim_{k}": v for k, v in analysis_obj.cost_details["latex_mapping"].items()
    }
    for curr_field in list(experiment_setting_details["node_classification"].keys()):
        if len(sim_cost_parameters) == 2:
            plot_heat_map_for_simulated(
                sum_clicks,
                curr_field,
                *sim_cost_parameters,
                *[sim_latex_mapping[parameter] for parameter in sim_cost_parameters],
            )
            plt.title(f"{curr_field.title()}")
            plt.savefig(
                subdirectory.joinpath(f"figs/optimal_{curr_field}.png"),
                bbox_inches="tight",
            )
        else:
            for subset in itertools.combinations(sim_cost_parameters, 2):
                plot_heat_map_for_simulated(
                    sum_clicks.groupby(list(subset), as_index=False).mean(),
                    curr_field,
                    *subset,
                    *[sim_latex_mapping[parameter] for parameter in subset],
                )
                plt.savefig(
                    subdirectory.joinpath(
                        f"figs/optimal_{curr_field}_{'_'.join(sorted(subset))}.png"
                    ),
                    bbox_inches="tight",
                )
