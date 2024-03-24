import logging
from argparse import ArgumentParser
from pathlib import Path

import dill as pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import yaml
from costometer.utils import AnalysisObject, set_plotting_and_logging_defaults
from sklearn.preprocessing import StandardScaler


def plot_regression_results(
    single_regressions, title, pretty_dependent_var, dependent_var, fig_path=None
):
    if not fig_path:
        fig_path = Path(__file__).resolve()

    plt.figure(figsize=(8, 6))

    ax = sns.barplot(
        x="Beta",
        y="Survey",
        data=single_regressions,
        linewidth=0,
        facecolor=(1, 1, 1, 0),
        errcolor=".2",
        edgecolor=".2",
    )

    plt.xticks(rotation=90)
    plt.title(f"{title}: {pretty_dependent_var}")
    ax.errorbar(
        single_regressions["Beta"].values,
        single_regressions.index,
        xerr=single_regressions["Standard error"].values,
        marker="o",
        mfc="black",
        mec="white",
        ms=10,
        mew=2,
        linewidth=0,
        elinewidth=1,
        ecolor=[
            "purple" if rsquared else "olive"
            for rsquared in single_regressions["Rsquared"]
        ],
    )

    twin_ax = ax.twinx()
    twin_ax.set_yticks(
        single_regressions.index,
        [f"(n={int(nobs)})" for nobs in single_regressions["Nobs"].values],
        style="italic",
    )
    twin_ax.set_ylim(ax.get_ylim())
    twin_ax.tick_params(axis="both", which="both", length=0)

    plt.savefig(
        fig_path.joinpath(
            f"figs/{title.replace(' ', '_').replace(':','')}_{dependent_var}.png"
        ),
        bbox_inches="tight",
    )


def f_square(r_squared_full, r_squared_base):
    return (r_squared_full - r_squared_base) / (1 - r_squared_full)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp",
        default="QuestPilot",
        dest="experiment_name",
    )
    parser.add_argument(
        "-a",
        "--analysis-file-name",
        default="pilot",
        dest="analysis_file_name",
    )
    parser.add_argument(
        "-s",
        "--subdirectory",
        default="questionnaire",
        dest="experiment_subdirectory",
        metavar="experiment_subdirectory",
    )
    inputs = parser.parse_args()

    subdirectory = Path(__file__).resolve().parents[1]
    irl_path = Path(__file__).resolve().parents[3]
    set_plotting_and_logging_defaults(
        subdirectory=subdirectory,
        experiment_name="AllVsTest",
        filename=Path(__file__).stem,
    )

    analysis_obj = AnalysisObject(
        inputs.experiment_name,
        irl_path=irl_path,
        experiment_subdirectory=inputs.experiment_subdirectory,
    )
    optimization_data = analysis_obj.query_optimization_data(
        excluded_parameters=analysis_obj.analysis_details.excluded_parameters
    )

    analysis_file_path = irl_path.joinpath(
        f"analysis/questionnaire/inputs/analysis/{inputs.analysis_file_name}.yaml"
    )

    with open(analysis_file_path, "rb") as f:
        analysis_yaml = yaml.safe_load(f)
    model_parameters = list(analysis_obj.cost_details.constant_values)

    # load data
    combined_scores = analysis_obj.dfs["combined_scores"].copy(deep=True)

    combined_scores = combined_scores.reset_index().merge(
        optimization_data[model_parameters + ["trace_pid"]],
        left_on="pid",
        right_on="trace_pid",
    )

    nonnumeric_cols = ["pid", "gender", "colorblind", "session"]

    # add psychiatric factor scores
    factor_scores = pd.read_csv(
        irl_path.joinpath(
            f"analysis/questionnaire/data/{inputs.experiment_name}/"
            f"{analysis_obj.analysis_details.loadings}_scores.csv"
        )
    )
    combined_scores = combined_scores.merge(factor_scores)

    # TODO: refactor this out, better way to average_node_cost
    combined_scores["average_node_cost"] = combined_scores.apply(
        lambda row: np.mean(
            [row[param] for param in analysis_obj.cost_details.cost_parameter_args]
        ),
        axis=1,
    )
    combined_scores["ptp"] = combined_scores.apply(
        lambda row: np.mean(
            [row[param] for param in ["pptsr", "pptlr", "ppmsr", "ppmlr"]]
        ),
        axis=1,
    )

    # standardize
    # see here: https://stats.stackexchange.com/q/29781
    scale = StandardScaler()
    combined_scores[
        combined_scores.columns.difference(nonnumeric_cols)
    ] = scale.fit_transform(
        combined_scores[combined_scores.columns.difference(nonnumeric_cols)]
    )

    bonferroni_corrected_pval = 0.05 / len(analysis_yaml["tests"]["regressions"])

    irl_path.joinpath("analysis/questionnaire/data/regressions/").mkdir(
        parents=True, exist_ok=True
    )

    for test in analysis_yaml["tests"]["regressions"]:
        if test["dependent"] in model_parameters:
            curr_cost_parameters = list(
                set(model_parameters) - set([test["dependent"]])
            )
            full_regression_formula = (
                f"{test['dependent']} ~ {test['independent']} + "
                + " + ".join(test["covariates"] + curr_cost_parameters + ["1"])
            )
            base_regression_formula = f"{test['dependent']} ~ " + " + ".join(
                test["covariates"] + curr_cost_parameters + ["1"]
            )
        else:
            full_regression_formula = (
                f"{test['dependent']} ~ {test['independent']} + "
                + " + ".join(test["covariates"] + ["1"])
            )
            base_regression_formula = f"{test['dependent']} ~ " + " + ".join(
                test["covariates"] + ["1"]
            )

        full_res = smf.ols(formula=full_regression_formula, data=combined_scores).fit(
            missing="drop"
        )
        base_res = smf.ols(formula=base_regression_formula, data=combined_scores).fit(
            missing="drop"
        )

        if full_res.pvalues[test["independent"]] <= bonferroni_corrected_pval:
            logging.info(full_regression_formula)
            logging.info(full_res.pvalues[test["independent"]])
            logging.info(test["independent"])
            logging.info("F-squared: {f_square(full_res.rsquared, base_res.rsquared)}")
            logging.info("F-squared: {f_square(full_res.rsquared, 0)}")
            logging.info(full_res.summary())

        with open(
            irl_path.joinpath(
                f"analysis/questionnaire/data/regressions/{full_regression_formula}.p"
            ),
            "wb",
        ) as f:
            pickle.dump(full_res, f)
