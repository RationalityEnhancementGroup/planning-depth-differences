import itertools
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
import statsmodels.formula.api as smf
from costometer.utils import (
    AnalysisObject,
    get_correlation_text,
    get_regression_text,
    set_font_sizes,
)

set_font_sizes()


def plot_simulated_recovery(mle):
    sim_cols = [col for col in list(mle) if "sim_" in col]
    plt.figure(figsize=(11.7, 8.27))
    melted_mle = pd.melt(
        mle,
        id_vars=sim_cols,
        value_vars=["given_cost", "depth_cost_weight"],
        ignore_index=False,
    ).reset_index()
    melted_mle["inferred"] = melted_mle.value
    melted_mle["simulated"] = melted_mle.apply(
        lambda row: row["sim_" + row["variable"]], axis=1
    )
    pretty_names = {
        "given_cost": "Effort Cost",
        "depth_cost_weight": "Planning Depth",
    }
    melted_mle["variable"] = melted_mle["variable"].apply(lambda var: pretty_names[var])
    ax = sns.pointplot(y="inferred", x="simulated", hue="variable", data=melted_mle)
    ax.legend(title="Cost Parameter")
    plt.xlabel("True parameter")
    plt.ylabel("Estimated parameter")


if __name__ == "__main__":
    """
    Example usage:
    python src/simulated_parameter_recovery.py -e SoftmaxRecovery
    """
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp",
        dest="experiment_name",
    )
    parser.add_argument(
        "-s",
        "--subdirectory",
        dest="experiment_subdirectory",
        metavar="experiment_subdirectory",
    )
    inputs = parser.parse_args()

    irl_path = Path(__file__).resolve().parents[4]
    data_path = irl_path.joinpath(f"analysis/{inputs.experiment_subdirectory}")

    analysis_obj = AnalysisObject(
        inputs.experiment_name,
        irl_path=irl_path,
        experiment_subdirectory=inputs.experiment_subdirectory,
    )

    optimization_data = analysis_obj.query_optimization_data()
    optimization_data = optimization_data[
        optimization_data["Model Name"] == "Effort Cost and Planning Depth"
    ]

    print("==========")
    for subset in itertools.combinations(
        ["given_cost", "depth_cost_weight", "temp"], 2
    ):
        correlation_object = pg.corr(
            optimization_data[subset[0]],
            optimization_data[subset[1]],
        )
        print("----------")
        print(
            f"Correlation between inferred {subset[0]} and {subset[1]} "
            f"for simulated participants"
        )
        print("----------")
        print(get_correlation_text(correlation_object))

    if "level_0" in list(optimization_data):
        del optimization_data["level_0"]
    if "Unnamed: 0" in list(optimization_data):
        del optimization_data["Unnamed: 0"]

    if "sim_temp" in optimization_data:
        model_params = analysis_obj.cost_details["dist_depth_eff_forw"][
            "cost_parameter_args"
        ] + ["temp"]
    else:
        model_params = analysis_obj.cost_details["dist_depth_eff_forw"][
            "cost_parameter_args"
        ]

    print("==========")
    for param in model_params:
        optimization_data[f"{param}_diff"] = (
            optimization_data[f"sim_{param}"] - optimization_data[param]
        )
        optimization_data[f"{param}_sq"] = optimization_data.apply(
            lambda row: row[f"{param}_diff"] ** 2, axis=1
        )
        print("----------")
        print(f"RMSE for parameter {param}")
        print("----------")
        print(
            np.sqrt(
                np.sum(optimization_data[f"{param}_sq"])
                / len(optimization_data[f"{param}_sq"])
            ),
        )
        optimization_data[f"{param}_rmse"] = np.sqrt(optimization_data[f"{param}_sq"])

    melted_rmse_df = pd.melt(
        optimization_data.reset_index(),
        value_vars=[
            f"{cost_param}_rmse"
            for cost_param in analysis_obj.cost_details["dist_depth_eff_forw"][
                "cost_parameter_args"
            ]
        ],
        id_vars="temp",
    )

    # optimization_data.groupby(["sim_temp"]).count()
    plt.figure(figsize=(11.7, 8.27))
    sns.pointplot(x="temp", y="value", hue="variable", data=melted_rmse_df)
    plt.xlabel("Simulated Agent Temperature")
    plt.ylabel("Simulated Agent RMSE")
    plt.tight_layout()
    plt.show()

    if "sim_temp" in optimization_data:
        plt.figure(figsize=(11.7, 8.27))
        ax = sns.pointplot(x="sim_temp", y="temp", data=optimization_data)
        plt.xlabel("Simulated Agent Temperature")
        plt.ylabel("Recovered Temperature (Log Scale)")
        ax.set_yscale("log")

    plt.tight_layout()
    plt.show()
    latex_names = {
        "given_cost": "\\costweight",
        "depth_cost_weight": "\\depthweight",
        "Intercept": "Intercept",
        "temp": "\\beta",
    }
    for param in model_params:
        print("==========")
        print(f"Regression with {param} as dependent variable")
        print("----------")
        mod = smf.ols(
            formula=f"sim_{param}  ~ given_cost " f"+ depth_cost_weight + " "temp + 1",
            data=optimization_data,
        )
        res = mod.fit()

        df_for_table = pd.DataFrame(
            {"coeff": res.params, "se": res.bse, "p": res.pvalues, "t": res.tvalues}
        )

        for row_idx, row in df_for_table.iterrows():
            if row["p"] > 0.05:
                pval_string = ""
            elif row["p"] < 0.001:
                pval_string = "^{***}"
            elif row["p"] < 0.01:
                pval_string = "^{**}"
            else:
                pval_string = "^{*}"

            print(
                f"${latex_names[param]}$  &  $\hat{{{latex_names[row_idx]}}}$ & "  # noqa: W605, E501
                f"${row['coeff']:.3f} ({row['se']:.3f}){pval_string}$ & ${row['t']:.3f}"
                f"$ \\\\"
            )

        print("----------")

        print(get_regression_text(res))

    for param in model_params:
        optimization_data[f"error_{param}"] = optimization_data.apply(
            lambda row: row[param] - row["sim_" + param], axis=1
        )
        print(f"Absolute error for {param}")
        print(
            f"Median: {optimization_data[f'error_{param}'].median():.2f}, "
            f"Range: [{optimization_data[f'error_{param}'].min():.2f}, "
            f"{optimization_data[f'error_{param}'].max():.2f}]"
        )

    plot_simulated_recovery(optimization_data)
    plt.savefig(
        data_path.joinpath(f"figs/{inputs.experiment_name}_parameter_recovery.png")
    )
