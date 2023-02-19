from argparse import ArgumentParser
from pathlib import Path

from costometer.utils import AnalysisObject
from quest_utils.factor_utils import get_psychiatric_scores, load_weights

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp",
        default="QuestPilot",
        dest="experiment_name",
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

    weights = load_weights(
        data_path.joinpath(f"inputs/loadings/{analysis_obj.loadings}.csv"),
        wise_weights=True if "Wise" in analysis_obj.loadings else False,
    )

    individual_items = analysis_obj.dfs["individual_items"]
    individual_items = individual_items[
        individual_items.columns.difference(["gender", "session"])
    ]

    scores = get_psychiatric_scores(individual_items, weights, scale_cols=True)

    data_path.joinpath(f"data/{inputs.experiment_name}").mkdir(
        parents=True, exist_ok=True
    )
    scores.to_csv(
        data_path.joinpath(
            f"data/{inputs.experiment_name}/{analysis_obj.loadings}_scores.csv"
        )
    )
