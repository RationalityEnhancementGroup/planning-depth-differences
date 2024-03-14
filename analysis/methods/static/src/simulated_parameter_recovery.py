import itertools
import logging
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
    get_pval_string,
    get_regression_text,
    set_font_sizes,
)

set_font_sizes()


def plot_simulated_recovery(mle, pretty_name_mapping):
    sim_cols = [col for col in list(mle) if "sim_" in col]
    plt.figure(figsize=(11.7, 8.27))
    melted_mle = pd.melt(
        mle,
        id_vars=sim_cols,
        value_vars=list(pretty_name_mapping.keys()),
        ignore_index=False,
    ).reset_index()
    melted_mle["inferred"] = melted_mle.value
    melted_mle["simulated"] = melted_mle.apply(
        lambda row: row["sim_" + row["variable"]], axis=1
    )

    melted_mle["variable"] = melted_mle["variable"].apply(
        lambda var: pretty_name_mapping[var]
    )
    ax = sns.pointplot(y="inferred", x="simulated", hue="variable", data=melted_mle)
    ax.legend(title="Parameter")
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
        default="methods/static",
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

    optimization_data = analysis_obj.query_optimization_data(
        excluded_parameters=analysis_obj.excluded_parameters
    )

    optimization_data = analysis_obj.query_optimization_data(
        excluded_parameters=analysis_obj.excluded_parameters
    )
    if analysis_obj.positive:
        optimization_data = optimization_data[
            (optimization_data["sim_depth_cost_weight"] >= 0)
            & (optimization_data["sim_given_cost"] >= 0)
        ]

    if analysis_obj.excluded_parameters == "":
        model = tuple()
    else:
        model = tuple(sorted(analysis_obj.excluded_parameters.split(",")))

    model_params = set(analysis_obj.cost_details["constant_values"]) - set(
        analysis_obj.excluded_parameters.split(",")
    )

    logging.info("==========")
    for subset in itertools.combinations(model_params, 2):
        correlation_object = pg.corr(
            optimization_data[subset[0]],
            optimization_data[subset[1]],
            method="spearman",
        )
        logging.info("----------")
        logging.info(
            f"Correlation between inferred {subset[0]} and inferred {subset[1]} "
            f"for simulated participants"
        )
        logging.info("----------")
        logging.info(get_correlation_text(correlation_object))

    plt.figure(figsize=(11.7, 8.27))
    latex_mapping = {
        **{
            f"sim_{k}": "$" + v + "$"
            for k, v in analysis_obj.cost_details["latex_mapping"].items()
        },
        **{
            k: "$\widehat{" + v + "}$"  # noqa: W605
            for k, v in analysis_obj.cost_details["latex_mapping"].items()
        },
    }
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        cmap=cmap,
        data=optimization_data[
            list(model_params) + [f"sim_{model_param}" for model_param in model_params]
        ]
        .rename(columns=latex_mapping)
        .corr("spearman"),
        annot=True,
        fmt=".2f",
    )
    plt.title(analysis_obj.model_name_mapping[model])
    plt.tight_layout()
    plt.savefig(
        data_path.joinpath(
            f"figs/{inputs.experiment_name}_parameter_recovery_correlation.png"
        )
    )

    logging.info("==========")
    for model_param in model_params:
        correlation_object = pg.corr(
            optimization_data[model_param],
            optimization_data[f"sim_{model_param}"],
            method="spearman",
        )
        logging.info("----------")
        logging.info(
            f"Correlation between inferred and actual {model_param} "
            f"for simulated participants"
        )
        logging.info("----------")
        logging.info(get_correlation_text(correlation_object))

    logging.info("==========")
    for param in model_params:
        optimization_data[f"{param}_diff"] = (
            optimization_data[f"sim_{param}"] - optimization_data[param]
        )
        optimization_data[f"{param}_sq"] = optimization_data.apply(
            lambda row: row[f"{param}_diff"] ** 2, axis=1
        )
        logging.info("----------")
        logging.info(f"RMSE for parameter {param}")
        logging.info("----------")
        logging.info(
            np.sqrt(
                np.sum(optimization_data[f"{param}_sq"])
                / len(optimization_data[f"{param}_sq"])
            ),
        )
        optimization_data[f"{param}_rmse"] = np.sqrt(optimization_data[f"{param}_sq"])

    melted_rmse_df = pd.melt(
        optimization_data.reset_index(),
        value_vars=[f"{param}_rmse" for param in model_params],
        id_vars="temp",
    )

    # optimization_data.groupby(["sim_temp"]).count()
    plt.figure(figsize=(11.7, 8.27))
    sns.pointplot(x="temp", y="value", hue="variable", data=melted_rmse_df)
    plt.xlabel("Simulated Agent Temperature")
    plt.ylabel("Simulated Agent RMSE")
    plt.tight_layout()
    plt.savefig(data_path.joinpath(f"figs/{inputs.experiment_name}_temp_vs_rmse.png"))

    if "sim_temp" in optimization_data:
        plt.figure(figsize=(11.7, 8.27))
        ax = sns.pointplot(x="sim_temp", y="temp", data=optimization_data)
        plt.xlabel("Simulated Agent Temperature")
        plt.ylabel("Recovered Temperature (Log Scale)")
        ax.set_yscale("log")
        plt.tight_layout()
        plt.savefig(
            data_path.joinpath(f"figs/{inputs.experiment_name}_recovered_temp.png")
        )

    for param in model_params:
        logging.info("==========")
        logging.info(f"Regression with {param} as dependent variable")
        logging.info("----------")
        mod = smf.ols(
            formula=f"sim_{param}  ~ {' + '.join(model_params)} + 1",
            data=optimization_data,
        )
        res = mod.fit()

        logging.info(param, res.rsquared.round(2))

        df_for_table = pd.DataFrame(
            {"coeff": res.params, "se": res.bse, "p": res.pvalues, "t": res.tvalues}
        )

        analysis_obj.cost_details["latex_mapping"]["Intercept"] = "\\text{Intercept}"
        for row_idx, row in df_for_table.iterrows():
            pval_string = get_pval_string(row["p"])

            logging.info(
                f"${analysis_obj.cost_details['latex_mapping'][param]}$  &  $\hat{{{analysis_obj.cost_details['latex_mapping'][row_idx]}}}$ & "  # noqa: W605, E501
                f"${row['coeff']:.3f} ({row['se']:.3f}){pval_string}$ & ${row['t']:.3f}"
                f"$ \\\\"
            )

        logging.info("----------")

        logging.info(get_regression_text(res))

    for param in model_params:
        optimization_data[f"error_{param}"] = optimization_data.apply(
            lambda row: row[param] - row["sim_" + param], axis=1
        )
        logging.info(f"Absolute error for {param}")
        logging.info(
            f"Median: {optimization_data[f'error_{param}'].median():.2f}, "
            f"Range: [{optimization_data[f'error_{param}'].min():.2f}, "
            f"{optimization_data[f'error_{param}'].max():.2f}]"
        )

    pretty_cost_names = dict(
        zip(
            analysis_obj.cost_details["cost_parameter_args"],
            analysis_obj.cost_details["cost_parameter_names"],
        )
    )
    pretty_names = {
        model_param: pretty_cost_names[model_param]
        if model_param in pretty_cost_names
        else model_param.title()
        for model_param in model_params
    }

    plot_simulated_recovery(
        optimization_data,
        {key: val for key, val in pretty_names.items() if key in pretty_cost_names},
    )
    plt.savefig(
        data_path.joinpath(f"figs/{inputs.experiment_name}_cost_parameter_recovery.png")
    )

    plot_simulated_recovery(
        optimization_data,
        {key: val for key, val in pretty_names.items() if key not in pretty_cost_names},
    )
    plt.savefig(
        data_path.joinpath(
            f"figs/{inputs.experiment_name}_additional_parameter_recovery.png"
        )
    )
