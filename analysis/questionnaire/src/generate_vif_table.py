from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import yaml
from costometer.utils import AnalysisObject
from quest_utils.analysis_utils import calculate_vif
from sklearn.preprocessing import StandardScaler

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
    combined_scores = combined_scores.merge(
        analysis_obj.dfs["quiz-and-demo"][["pid", "age", "gender"]]
    )

    # standardize
    # see here: https://stats.stackexchange.com/q/29781
    scale = StandardScaler()
    combined_scores[
        combined_scores.columns.difference(nonnumeric_cols)
    ] = scale.fit_transform(
        combined_scores[combined_scores.columns.difference(nonnumeric_cols)]
    )

    vif_dict = calculate_vif(
        exogenous_df=combined_scores[
            [
                model_parameter
                for model_parameter in model_parameters
                if len(combined_scores[model_parameter].unique()) > 1
            ]
            + ["gender", "age", "IQ"]
        ]
    )

    analysis_obj.cost_details["latex_mapping"] = {
        **{
            f"${key}$": val
            for key, val in analysis_obj.cost_details["latex_mapping"].items()
        },
        "const": "Constant",
        "female": "Gender: Woman",
        "other": "Gender: Non-binary or did not disclose",
        "age": "Age",
    }

    for variable, vif_value in vif_dict.items():
        print(
            f"{analysis_obj.cost_details['latex_mapping'][variable]  if variable in analysis_obj.cost_details['latex_mapping'] else variable} & {vif_value:.2f} \\\ "
        )  # noqa
