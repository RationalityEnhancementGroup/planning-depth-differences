import itertools
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
import statsmodels.formula.api as smf
from costometer.utils import (
    get_correlation_text,
    get_pval_string,
    get_regression_text,
    standard_parse_args,
)


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
    irl_path = Path(__file__).resolve().parents[4]
    analysis_obj, inputs, subdirectory = standard_parse_args(
        description=sys.modules[__name__].__doc__,
        irl_path=irl_path,
        filename=Path(__file__).stem,
        default_experiment="SoftmaxRecovery",
    )

    optimization_data = analysis_obj.query_optimization_data(
        excluded_parameters=analysis_obj.analysis_details.excluded_parameters
    )

    optimization_data = analysis_obj.query_optimization_data(
        excluded_parameters=analysis_obj.analysis_details.excluded_parameters
    )

    model = analysis_obj.analysis_details.excluded_parameters

    model_params = set(analysis_obj.cost_details.constant_values) - set(
        analysis_obj.analysis_details.excluded_parameters
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
            "Correlation between inferred %s and "
            "inferred %s for simulated participants",
            subset[0],
            subset[1],
        )
        logging.info("----------")
        logging.info(get_correlation_text(correlation_object))

    plt.figure(figsize=(11.7, 8.27))
    latex_mapping = {
        **{
            f"sim_{k}": "$" + v + "$"
            for k, v in analysis_obj.cost_details.latex_mapping.items()
        },
        **{
            k: "$\widehat{" + v + "}$"  # noqa: W605
            for k, v in analysis_obj.cost_details.latex_mapping.items()
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
        subdirectory.joinpath(
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
            "Correlation between inferred and actual %s for simulated participants",
            model_param,
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
        logging.info("RMSE for parameter {param}")
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
    plt.savefig(
        subdirectory.joinpath(f"figs/{inputs.experiment_name}_temp_vs_rmse.png")
    )

    if "sim_temp" in optimization_data:
        plt.figure(figsize=(11.7, 8.27))
        ax = sns.pointplot(x="sim_temp", y="temp", data=optimization_data)
        plt.xlabel("Simulated Agent Temperature")
        plt.ylabel("Recovered Temperature (Log Scale)")
        ax.set_yscale("log")
        plt.tight_layout()
        plt.savefig(
            subdirectory.joinpath(f"figs/{inputs.experiment_name}_recovered_temp.png")
        )

    for param in model_params:
        logging.info("==========")
        logging.info("Regression with {param} as dependent variable")
        logging.info("----------")
        mod = smf.ols(
            formula=f"sim_{param}  ~ {' + '.join(model_params)} + 1",
            data=optimization_data,
        )
        res = mod.fit()

        logging.info("%s: %.2f", param, res.rsquared)

        df_for_table = pd.DataFrame(
            {"coeff": res.params, "se": res.bse, "p": res.pvalues, "t": res.tvalues}
        )

        analysis_obj.cost_details.latex_mapping["Intercept"] = "\\text{Intercept}"
        for row_idx, row in df_for_table.iterrows():
            pval_string = get_pval_string(row["p"])

            logging.info(
                "$%s$  &  $\hat{%s}$ & $%.3f (%.3f)%s$ & $%.3f$ \\",  # noqa: W605
                analysis_obj.cost_details.latex_mapping[param],
                analysis_obj.cost_details.latex_mapping[row_idx],
                row["coeff"],
                row["se"],
                pval_string,
                row["t"],
            )

        logging.info("----------")

        logging.info(get_regression_text(res))

    for param in model_params:
        optimization_data[f"error_{param}"] = optimization_data.apply(
            lambda row: row[param] - row["sim_" + param], axis=1
        )
        logging.info("Absolute error for {param}")
        logging.info(
            "Median: %.2f, Range: [%.2f, %.2f]",
            optimization_data[f"error_{param}"].median(),
            optimization_data[f"error_{param}"].min(),
            optimization_data[f"error_{param}"].max(),
        )

    pretty_cost_names = dict(
        zip(
            analysis_obj.cost_details.cost_parameter_args,
            analysis_obj.cost_details.cost_parameter_names,
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
        subdirectory.joinpath(
            f"figs/{inputs.experiment_name}_cost_parameter_recovery.png"
        )
    )

    plot_simulated_recovery(
        optimization_data,
        {key: val for key, val in pretty_names.items() if key not in pretty_cost_names},
    )
    plt.savefig(
        subdirectory.joinpath(
            f"figs/{inputs.experiment_name}_additional_parameter_recovery.png"
        )
    )
