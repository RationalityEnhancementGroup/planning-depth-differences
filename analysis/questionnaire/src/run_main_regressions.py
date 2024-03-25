import logging
from argparse import ArgumentParser
from pathlib import Path

import dill as pickle
import pandas as pd
import statsmodels.formula.api as smf
import yaml
from costometer.utils import (
    AnalysisObject,
    get_parameter_coefficient,
    get_pval_text,
    get_regression_text,
    set_plotting_and_logging_defaults,
)
from quest_utils.subscale_utils import uppsp_dict
from scipy.stats import mode
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests


def f_square(r_squared_full, r_squared_base):
    return (r_squared_full - r_squared_base) / (1 - r_squared_full)


def get_data_subset(test, data, params):
    """
    Drop any columns with nans for full model.

    (Reason: statsmodels defaults to wrong degrees of freedom numerator
    with nans only in full model in some cases.)
    """
    cols = (
        [
            model_parameter
            for model_parameter in params
            if len(data[model_parameter].unique()) > 1
        ]
        + [
            covariate.replace("C(", "").replace(")", "")
            for covariate in test["covariates"]
        ]
        + [test["dependent"]]
    )

    return data[cols].copy(deep=True).dropna(axis=0)


def run_regression(test, data, params):
    independent_variables = (
        [
            model_parameter
            for model_parameter in params
            if len(data[model_parameter].unique()) > 1
        ]
        + test["covariates"]
        + ["1"]
    )

    if len(independent_variables) > 1:
        full_regression_formula = f"Q('{test['dependent']}') ~ " + " + ".join(
            independent_variables
        )
    else:
        full_regression_formula = (
            f"Q('{test['dependent']}') ~ {independent_variables[0]}"
        )

    full_res = smf.ols(formula=full_regression_formula, data=data).fit()
    return full_res


def run_main_regressions(tests, combined_scores, model_parameters, pval_cutoff=0.05):
    # add all results to this
    pvals = []
    results = []

    for test in tests:
        curr_data = get_data_subset(test, data=combined_scores, params=model_parameters)

        full_res = run_regression(test, data=curr_data, params=model_parameters)
        base_res = run_regression(test, data=curr_data, params=[])

        res = anova_lm(base_res, full_res)
        res["prettyname"] = test["prettyname"]
        res["fsquare"] = f_square(full_res.rsquared, base_res.rsquared)

        pvals.append(res.loc[1]["Pr(>F)"])
        results.append(res)

    # correct p values
    reject_null, corrected_pval, _, _ = multipletests(
        pvals, alpha=pval_cutoff, method="fdr_bh"
    )

    for test_idx, res in enumerate(results):
        logging.info(
            "%s & $%.3f$ & $F(%.0f, %.0f) = %.2f$ & %s\\ ",
            res["prettyname"].loc[1],
            res.loc[1]["fsquare"],
            res.loc[1]["df_diff"],
            res.loc[1]["df_resid"],
            res.loc[1]["F"],
            get_pval_text(corrected_pval[test_idx]),
        )

    for test_idx, test in enumerate(tests):
        curr_data = get_data_subset(test, data=combined_scores, params=model_parameters)

        full_res = run_regression(test, data=curr_data, params=model_parameters)

        if reject_null[test_idx]:
            logging.info(test["prettyname"])
            logging.info("\t - %s", get_regression_text(full_res))

            params_to_correct_for = [
                model_parameter
                for model_parameter in list(set(model_parameters) - {test["followup"]})
                if model_parameter in full_res.pvalues
            ]  # if no one classified as this strategy, won't be in model

            coeff_reject_null, coeff_corrected_pval, _, _ = multipletests(
                [
                    full_res.pvalues[model_parameter]
                    for model_parameter in params_to_correct_for
                ],
                alpha=pval_cutoff,
                method="fdr_bh",
            )

            # if followup, correct
            if len(test["followup"]) > 0:
                if full_res.pvalues[test["followup"]] < pval_cutoff:
                    logging.info(
                        "\t\t - %s, %s",
                        test["followup"],
                        get_parameter_coefficient(
                            full_res,
                            test["followup"],
                            pval=full_res.pvalues[test["followup"]],
                        ),
                    )

            for param_idx, curr_param in enumerate(params_to_correct_for):
                if coeff_reject_null[param_idx]:
                    logging.info(
                        "\t\t - %s, %s",
                        curr_param,
                        get_parameter_coefficient(
                            full_res, curr_param, pval=coeff_corrected_pval[param_idx]
                        ),
                    )

    return pvals


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp",
        default="QuestMain",
        dest="experiment_name",
    )
    parser.add_argument(
        "-a",
        "--analysis-file-name",
        default="main_experiment",
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
        experiment_name="MainRegressions",
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

    model_parameters = list(
        set(analysis_obj.cost_details.constant_values)
        - set(analysis_obj.analysis_details.excluded_parameters)
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
            f"{analysis_obj.analysis_details.loadings}_scores.csv"
        )
    )
    combined_scores = combined_scores.merge(factor_scores)
    combined_scores = combined_scores.merge(
        analysis_obj.dfs["quiz-and-demo"][["pid", "age", "gender"]]
    )

    combined_scores = combined_scores.merge(
        analysis_obj.dfs["mouselab-mdp"]
        .groupby(["pid"], as_index=False)
        .mean()[["pid", "num_early", "num_middle", "num_late", "num_clicks"]],
        on="pid",
    )

    for upps_subscale in set(uppsp_dict.values()):
        combined_scores[upps_subscale] = combined_scores["pid"].apply(
            lambda curr_pid: sum(
                [
                    individual_items.loc[curr_pid][key]
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

    if "clicks" in inputs.analysis_file_name:
        full_df = run_main_regressions(
            analysis_yaml["regressions"][0]["tests"],
            combined_scores,
            ["num_early", "num_middle", "num_late"],
            pval_cutoff=0.05,
        )
    elif "cm" in inputs.analysis_file_name:
        cm = {}
        for session in analysis_obj.sessions:
            with open(irl_path.joinpath(f"cluster/data/cm/{session}.pkl"), "rb") as f:
                exp = pickle.load(f)
            for pid, strategies in exp.participant_strategies.items():
                last_strategies = strategies[-20:]
                cm[pid] = mode(last_strategies).mode[0]

        combined_scores["mode_strategy"] = combined_scores["pid"].apply(
            lambda pid: cm[pid]
        )
        dummies = pd.get_dummies(combined_scores["mode_strategy"], prefix="cm")
        combined_scores = pd.concat([combined_scores, dummies], axis=1)

        full_df = run_main_regressions(
            analysis_yaml["regressions"][0]["tests"],
            combined_scores,
            list(dummies.columns.values),
            pval_cutoff=0.05,
        )
    elif "demo" in inputs.analysis_file_name:
        dummies = pd.get_dummies(combined_scores["gender"], prefix="gender")
        combined_scores = pd.concat([combined_scores, dummies], axis=1)

        full_df = run_main_regressions(
            analysis_yaml["regressions"][0]["tests"],
            combined_scores,
            ["age", "IQ"] + list(dummies.columns.values),
            pval_cutoff=0.05,
        )
    else:
        full_df = run_main_regressions(
            analysis_yaml["regressions"][0]["tests"],
            combined_scores,
            model_parameters,
            pval_cutoff=0.05,
        )
