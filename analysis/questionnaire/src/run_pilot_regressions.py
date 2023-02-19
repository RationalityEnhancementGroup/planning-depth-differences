from argparse import ArgumentParser
from pathlib import Path

import dill as pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import yaml
from costometer.utils import AnalysisObject, set_font_sizes
from quest_utils.factor_utils import col_dict, load_weights
from run_regressions import plot_regression_results
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

set_font_sizes()

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
    model_parameters = list(analysis_obj.cost_details["constant_values"])

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

    # STAI and OCIR have high enough R^2
    for curr_scale in ["anxiety", "ocir"]:
        weights = load_weights(
            data_path.joinpath(f"inputs/loadings/{analysis_obj.loadings}.csv"),
            wise_weights=True if "Wise" in analysis_obj.loadings else False,
        )
        reduced_items = [
            item.replace(f"{col_dict[curr_scale]}_", f"{curr_scale}_").replace("_", ".")
            for item in weights.index.values
            if f"{col_dict[curr_scale]}_" in item
        ]
        combined_scores[col_dict[curr_scale]] = combined_scores["pid"].apply(
            lambda pid: individual_items[reduced_items].sum(axis=1)[pid]
        )

    # standardize
    # see here: https://stats.stackexchange.com/q/29781
    scale = StandardScaler()
    combined_scores[
        combined_scores.columns.difference(nonnumeric_cols)
    ] = scale.fit_transform(
        combined_scores[combined_scores.columns.difference(nonnumeric_cols)]
    )

    bonferroni_corrected_pval = 0.05

    # add all results to this
    full_df = []

    for dependent_variable in [
        "STAI",
        "OCIR",
        "Q('UPPS-P')",
        "DOSPERT",
        "FTP",
        "crt",
        "SW",
        "CIT",
        "AD",
        "Q('life-regrets')",
        "ppmlr",
        "ppmsr",
        "pptlr",
        "pptsr",
        "ptp",
        "regrets",
        "satisfaction",
    ]:
        test = {
            "dependent": dependent_variable,
            "covariates": ["age", "IQ", "C(gender)"],
        }

        full_regression_formula = f"{dependent_variable} ~ " + " + ".join(
            [
                model_parameter
                for model_parameter in model_parameters
                if len(combined_scores[model_parameter].unique()) > 1
            ]
            + test["covariates"]
            + ["temp", "1"]
        )

        try:
            full_res = smf.ols(
                formula=full_regression_formula, data=combined_scores
            ).fit(missing="drop")

            print(full_res.summary())
            category_data = []
            for independent_variable in model_parameters:

                # all undirected tests
                category_data.append(
                    [
                        full_res.bse[independent_variable],
                        full_res.params[independent_variable],
                        dependent_variable,
                        full_res.pvalues[independent_variable],
                        independent_variable,
                        full_res.nobs,
                        True
                        if full_res.rsquared > 0.1 and full_res.f_pvalue < 0.05
                        else False,
                    ]
                )

            irl_path.joinpath(
                f"analysis/questionnaire/data/regressions/{inputs.analysis_file_name}"
            ).mkdir(parents=True, exist_ok=True)
            with open(
                irl_path.joinpath(
                    f"analysis/questionnaire/data/regressions/"
                    f"{inputs.analysis_file_name}/{full_regression_formula}.p"
                ),
                "wb",
            ) as f:
                pickle.dump(full_res, f)
        except:  # noqa : E722
            print(dependent_variable, independent_variable)

        category_df = pd.DataFrame(
            category_data,
            columns=[
                "Standard error",
                "Beta",
                "Survey",
                "Pval",
                "Dependent",
                "Nobs",
                "Rsquared",
            ],
        )
        category_df["Category title"] = dependent_variable.title()
        full_df.append(category_df)

    full_df = pd.concat(full_df)
    reject, pvals_corrected, _, _ = multipletests(
        full_df["Pval"].values, method="fdr_bh", alpha=0.05
    )
    full_df["reject"] = reject
    full_df["Survey"] = full_df.apply(
        lambda row: f"{row['Survey']}{' *' if row['reject'] else ''}", axis=1
    )

    for dependent_var in full_df["Dependent"].unique():
        plot_df = (
            full_df[full_df["Dependent"] == dependent_var]
            .sort_values(by=["Beta"])
            .reset_index(drop=True)
        )
        plt.figure()
        plot_regression_results(
            plot_df,
            "Pilot Regressions",
            analysis_obj.cost_details["latex_mapping"][dependent_var],
            dependent_var,
            fig_path=irl_path.joinpath("analysis/questionnaire"),
        )
        plt.tight_layout()
        plt.show()
