import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from costometer.utils import get_static_palette, standard_parse_args


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
    irl_path = Path(__file__).resolve().parents[4]
    analysis_obj, inputs, subdirectory = standard_parse_args(
        description=sys.modules[__name__].__doc__,
        irl_path=irl_path,
        filename=Path(__file__).stem,
    )

    optimization_data = analysis_obj.query_optimization_data(
        excluded_parameters=analysis_obj.analysis_details.excluded_parameters
    )

    plotting_df = []
    for param in analysis_obj.cost_details.cost_parameter_args:
        curr_df = pd.DataFrame(optimization_data[param].copy(deep=True))
        curr_df["Model Parameter"] = (
            "$" + analysis_obj.cost_details.latex_mapping[param] + "$"
        )
        curr_df = curr_df.rename(columns={param: "Parameter Value"})
        plotting_df.append(curr_df)

    # for analysis_obj.cost_details[]
    sns.boxplot(x="Model Parameter", y="Parameter Value", data=pd.concat(plotting_df))

    plt.savefig(f"figs/{inputs.experiment_name}_model_parameter_values.png")
