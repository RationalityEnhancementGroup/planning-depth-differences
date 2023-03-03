from argparse import ArgumentParser
from pathlib import Path

import dill as pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import yaml
from costometer.utils import (
    AnalysisObject,
    get_parameter_coefficient,
    get_pval_string,
    get_regression_text,
)
from quest_utils.subscale_utils import uppsp_dict
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


def calculate_vif(exogenous_df):
    exogenous_df = add_constant(exogenous_df)
    exogenous_df = exogenous_df.dropna()

    if "gender" in exogenous_df:
        exogenous_df = pd.concat(
            [exogenous_df, pd.get_dummies(exogenous_df["gender"])], axis=1
        )
        del exogenous_df["gender"]
        del exogenous_df["male"]

    return dict(
        zip(
            exogenous_df.columns,
            [
                variance_inflation_factor(exogenous_df.values, col_idx)
                for col_idx in range(len(exogenous_df.columns))
            ],
        )
    )


def plot_regression_results(regression_df, title, pretty_dependent_var_dict):

    f, ax = plt.subplots(2, 3, figsize=(50, 20), dpi=100, facecolor="white")

    for dependent_var_idx, dependent_var in enumerate(
        regression_df["Dependent"].unique()
    ):
        curr_subset = (
            regression_df[regression_df["Dependent"] == dependent_var]
            .sort_values(by="Coefficient")
            .reset_index(drop=True)
        )
        curr_ax = ax[np.unravel_index(dependent_var_idx, (2, 3))]
        sns.barplot(
            x="Coefficient",
            y="Survey / Task",
            data=curr_subset,
            linewidth=0,
            facecolor=(1, 1, 1, 0),
            errcolor=".2",
            edgecolor=".2",
            ax=curr_ax,
        )

        curr_ax.tick_params(axis="x", rotation=90)
        curr_ax.title.set_text(f"{title}: ${pretty_dependent_var_dict[dependent_var]}$")
        curr_ax.errorbar(
            curr_subset["Coefficient"].values,
            curr_subset.index,
            xerr=curr_subset["Standard error"].values,
            marker="o",
            mfc="black",
            mec="white",
            ms=10,
            mew=2,
            linewidth=0,
            elinewidth=1,
            ecolor=[
                "purple" if rsquared else "olive"
                for rsquared in curr_subset["Rsquared"]
            ],
        )

        twin_ax = curr_ax.twinx()
        twin_ax.set_yticks(
            curr_subset.index,
            [f"(n={int(nobs)})" for nobs in curr_subset["Nobs"].values],
            style="italic",
        )
        twin_ax.set_ylim(curr_ax.get_ylim())
        twin_ax.tick_params(axis="both", which="both", length=0)

    return f


def get_regression_df(
    tests, combined_scores, model_parameters, pval_cutoff=0.05, regression_path=None
):
    # add all results to this
    full_df = []

    for test in tests:
        full_regression_formula = f"{test['dependent']} ~ " + " + ".join(
            [
                model_parameter
                for model_parameter in model_parameters
                if len(combined_scores[model_parameter].unique()) > 1
            ]
            + test["covariates"]
            + ["1"]
        )

        for covariate in test["covariates"]:
            if ":" in covariate:
                combined_scores[covariate] = combined_scores.apply(
                    lambda row: np.prod([row[col] for col in covariate.split(":")]),
                    axis=1,
                )
        vif_dict = calculate_vif(
            exogenous_df=combined_scores[
                [
                    model_parameter
                    for model_parameter in model_parameters
                    if len(combined_scores[model_parameter].unique()) > 1
                ]
                + [
                    covariate.replace("C(", "").replace(")", "")
                    for covariate in test["covariates"]
                ]
            ]
        )
        print({key: val for key, val in vif_dict.items() if val >= 5})

        full_res = smf.ols(formula=full_regression_formula, data=combined_scores).fit(
            missing="drop"
        )

        print(test["dependent"])
        print("\t - " + get_regression_text(full_res))

        if full_res.f_pvalue < 0.05:
            for param in list(full_res.pvalues[(full_res.pvalues < pval_cutoff)].index):
                print(f"\t\t - {param}, {get_parameter_coefficient(full_res, param)}")

        for model_parameter in model_parameters:

            # all undirected tests
            full_df.append(
                [
                    full_res.bse[model_parameter],
                    full_res.params[model_parameter],
                    test["prettyname"],
                    full_res.pvalues[model_parameter],
                    model_parameter,
                    full_res.nobs,
                    True if full_res.f_pvalue < pval_cutoff else False,
                ]
            )

        if regression_path:
            regression_path.mkdir(parents=True, exist_ok=True)
            with open(
                regression_path.joinpath(f"{full_regression_formula}.p"),
                "wb",
            ) as f:
                pickle.dump(full_res, f)

    full_df = pd.DataFrame(
        full_df,
        columns=[
            "Standard error",
            "Coefficient",
            "Survey / Task",
            "Pval",
            "Dependent",
            "Nobs",
            "Rsquared",
        ],
    )

    full_df["Survey / Task"] = full_df.apply(
        lambda row: f"{row['Survey / Task']}"
        f"{'$'+get_pval_string(row['Pval'])+'$' if get_pval_string(row['Pval']) != '' else ''}",  # noqa : E501
        axis=1,
    )
    return full_df


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
        default="pilot_full_plot",
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

    data_path = Path(__file__).resolve().parents[1]
    irl_path = Path(__file__).resolve().parents[3]

    analysis_obj = AnalysisObject(
        inputs.experiment_name,
        irl_path=irl_path,
        experiment_subdirectory=inputs.experiment_subdirectory,
    )
    optimization_data = analysis_obj.query_optimization_data(
        excluded_parameters=analysis_obj.excluded_parameters
    )

    analysis_file_path = irl_path.joinpath(
        f"analysis/questionnaire/inputs/analysis/{inputs.analysis_file_name}.yaml"
    )

    with open(analysis_file_path, "rb") as f:
        analysis_yaml = yaml.safe_load(f)

    model_parameters = list(
        set(analysis_obj.cost_details["constant_values"])
        - set(analysis_obj.excluded_parameters.split(","))
    )

    # load data
    combined_scores = analysis_obj.dfs["combined_scores"].copy(deep=True)
    individual_items = analysis_obj.dfs["individual_items"].copy(deep=True)

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
            f"{analysis_obj.loadings}_scores.csv"
        )
    )
    combined_scores = combined_scores.merge(factor_scores)

    combined_scores["ptp"] = combined_scores.apply(
        lambda row: np.mean(
            [row[param] for param in ["pptsr", "pptlr", "ppmsr", "ppmlr"]]
        ),
        axis=1,
    )

    for upps_subscale in set(uppsp_dict.values()):
        combined_scores[upps_subscale] = combined_scores["pid"].apply(
            lambda pid: sum(
                [
                    individual_items.loc[pid][key]
                    for key, val in uppsp_dict.items()
                    if val == upps_subscale
                ]
            )
        )

    # standardize
    # see here: https://stats.stackexchange.com/q/29781
    scale = StandardScaler()
    combined_scores[
        combined_scores.columns.difference(nonnumeric_cols)
    ] = scale.fit_transform(
        combined_scores[combined_scores.columns.difference(nonnumeric_cols)]
    )

    full_df = get_regression_df(
        analysis_yaml["regressions"][0]["tests"],
        combined_scores,
        model_parameters,
        pval_cutoff=0.005,
        regression_path=irl_path.joinpath(
            f"analysis/questionnaire/data/regressions/{inputs.analysis_file_name}"
        ),
    )
    plot_regression_results(
        full_df,
        analysis_yaml["regressions"][0]["title"],
        analysis_obj.cost_details["latex_mapping"],
    )
    plt.tight_layout()
    plt.show()
