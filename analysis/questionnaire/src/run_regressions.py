from argparse import ArgumentParser
from pathlib import Path

import dill as pickle
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import yaml
from costometer.utils import AnalysisObject
from sklearn.preprocessing import StandardScaler


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

    irl_path = Path(__file__).resolve().parents[3]

    analysis_obj = AnalysisObject(
        inputs.experiment_name,
        irl_path=irl_path,
        experiment_subdirectory=inputs.experiment_subdirectory,
    )
    optimization_data = analysis_obj.query_optimization_data()

    analysis_file_path = irl_path.joinpath(
        f"analysis/questionnaire/inputs/analysis/{inputs.analysis_file_name}.yaml"
    )

    with open(analysis_file_path, "rb") as f:
        analysis_yaml = yaml.safe_load(f)

    # read in cost function details
    yaml_path = irl_path.joinpath(
        f"data/inputs/yamls/cost_functions/{analysis_yaml['cost_function']}.yaml"
    )

    with open(yaml_path, "r") as stream:
        cost_details = yaml.safe_load(stream)

    cost_parameters = cost_details["cost_parameter_args"] + ["temp"]

    # load data
    combined_scores = pd.concat(
        [
            pd.read_csv(
                irl_path.joinpath(f"data/processed/{session}/combined_scores.csv")
            )
            for session in analysis_yaml["data"]
        ]
    )
    optimization_data[
        optimization_data["Model Name"] == "Distance, Depth and Effort Costs"
    ]
    combined_scores = combined_scores.merge(
        optimization_data[cost_parameters + ["trace_pid"]],
        left_on="pid",
        right_on="trace_pid",
    )

    nonnumeric_cols = ["pid", "gender", "colorblind"]

    # add psychiatric factor scores
    factor_scores = pd.read_csv(
        irl_path.joinpath(
            f"analysis/questionnaire/data/{inputs.experiment_name}/scores.csv"
        )
    )
    combined_scores = combined_scores.merge(factor_scores)

    # TODO: refactor this out, better way to average_node_cost
    combined_scores["average_node_cost"] = combined_scores.apply(
        lambda row: np.mean(
            [row[param] for param in cost_details["cost_parameter_args"]]
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

    for test in analysis_yaml["tests"]["regressions"]:

        if test["dependent"] in cost_parameters:
            curr_cost_parameters = list(set(cost_parameters) - set([test["dependent"]]))
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
                + " + ".join(test["covariates"] + ["temp", "1"])
            )
            base_regression_formula = f"{test['dependent']} ~ " + " + ".join(
                test["covariates"] + ["temp", "1"]
            )

        full_res = smf.ols(formula=full_regression_formula, data=combined_scores).fit(
            missing="drop"
        )
        base_res = smf.ols(formula=base_regression_formula, data=combined_scores).fit(
            missing="drop"
        )

        if full_res.pvalues[test["independent"]] <= bonferroni_corrected_pval:
            print(full_regression_formula)
            print(test["independent"])
            print(f"F-squared: {f_square(full_res.rsquared, base_res.rsquared)}")
            print(f"F-squared: {f_square(full_res.rsquared, 0)}")
            print(full_res.summary())

        with open(
            irl_path.joinpath(
                f"analysis/questionnaire/data/regressions/{full_regression_formula}.p"
            ),
            "wb",
        ) as f:
            pickle.dump(full_res, f)
