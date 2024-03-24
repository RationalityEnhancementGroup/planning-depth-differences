import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from costometer.utils.scripting_utils import standard_parse_args
from statsmodels.tools.eval_measures import bic

if __name__ == "__main__":
    """
    Example usage:
    python src/model_recovery.py
    """
    irl_path = Path(__file__).resolve().parents[4]
    analysis_obj, inputs, subdirectory = standard_parse_args(
        description=sys.modules[__name__].__doc__,
        irl_path=irl_path,
        filename=Path(__file__).stem,
        default_experiment="SoftmaxRecovery",
    )

    optimization_data = analysis_obj.query_optimization_data()

    if analysis_obj.analysis_details.excluded_parameters == "":
        excluded_set = set()
    else:
        excluded_set = set(analysis_obj.analysis_details.excluded_parameters)

    optimization_data = optimization_data[
        optimization_data.apply(
            lambda row: excluded_set.issubset(row["model"])
            or (row["Model Name"] == "Null"),
            axis=1,
        )
    ]

    optimization_data = optimization_data[
        optimization_data["applied_policy"] != "RandomPolicy"
    ]
    # unfortunately don't have given cost 1
    analysis_obj.cost_details.constant_values["given_cost"] = 0

    full_df = []
    for model in (
        optimization_data["model"]
        .apply(lambda model: tuple(sorted(model)) if model != "None" else ())
        .unique()
    ):
        if model != "":
            curr_optimization_data = optimization_data[
                optimization_data.apply(
                    lambda row: np.all(
                        [
                            row[f"sim_{param}"]
                            == analysis_obj.cost_details.constant_values[param]
                            for param in model
                        ]
                    ),
                    axis=1,
                )
            ]
        else:
            curr_optimization_data = optimization_data

        bic_df = (
            curr_optimization_data.groupby(["Model Name", "Number Parameters"])
            .sum()
            .reset_index()
        )

        bic_df["bic"] = bic_df.apply(
            lambda row: bic(
                llf=row["mle"],
                nobs=row["num_clicks"],
                df_modelwc=row["Number Parameters"],
            ),
            axis=1,
        )

        curr_df = bic_df[["Model Name", "bic"]]
        curr_df["Simulated Model"] = analysis_obj.cost_details.model_name_mapping[model]
        full_df.append(curr_df)

    full_df = pd.concat(full_df)

    full_df = full_df[
        full_df.apply(
            lambda row: row["Model Name"] in full_df["Simulated Model"].unique(), axis=1
        )
    ]
    heat_map_data = full_df.pivot(
        index="Model Name", columns="Simulated Model", values="bic"
    )
    for col in heat_map_data.columns:
        heat_map_data[col] = (heat_map_data[col] - min(heat_map_data[col])) / (
            max(heat_map_data[col]) - min(heat_map_data[col])
        )

    plt.figure(figsize=(80, 60))
    sns.heatmap(data=heat_map_data, annot=True, fmt=".2f")

    plt.savefig(
        subdirectory.joinpath(f"figs/{inputs.experiment_name}_model_recovery.png"),
        bbox_inches="tight",
    )
