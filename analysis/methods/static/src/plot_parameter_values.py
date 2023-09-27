from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from costometer.utils import AnalysisObject, get_static_palette, set_font_sizes

set_font_sizes(SMALL_SIZE=14)

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
        data=optimization_data.sort_values(by="bic")[:33],
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
        default="MainExperiment",
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

    optimization_data = analysis_obj.query_optimization_data(
        excluded_parameters=analysis_obj.excluded_parameters
    )

    plotting_df = []
    for param in analysis_obj.cost_details["cost_parameter_args"]:
        curr_df = pd.DataFrame(optimization_data[param].copy(deep=True))
        curr_df["Model Parameter"] = (
            "$" + analysis_obj.cost_details["latex_mapping"][param] + "$"
        )
        curr_df = curr_df.rename(columns={param: "Parameter Value"})
        plotting_df.append(curr_df)

    # for analysis_obj.cost_details[]
    sns.boxplot(x="Model Parameter", y="Parameter Value", data=pd.concat(plotting_df))

    plt.show()
