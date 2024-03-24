import sys
from pathlib import Path

from costometer.utils import standard_parse_args
from quest_utils.factor_utils import get_psychiatric_scores, load_weights

if __name__ == "__main__":
    irl_path = Path(__file__).resolve().parents[3]
    analysis_obj, inputs, subdirectory = standard_parse_args(
        description=sys.modules[__name__].__doc__,
        irl_path=irl_path,
        filename=Path(__file__).stem,
        default_experiment="QuestMain",
        default_subdirectory="questionnaire",
    )

    weights = load_weights(
        subdirectory.joinpath(
            f"inputs/loadings/{analysis_obj.analysis_details.loadings}.csv"
        ),
        wise_weights=True
        if "Wise" in analysis_obj.analysis_details.loadings
        else False,
    )

    individual_items = analysis_obj.dfs["individual_items"]
    individual_items = individual_items[
        individual_items.columns.difference(["gender", "session"])
    ]

    scores = get_psychiatric_scores(individual_items, weights, scale_cols=True)

    subdirectory.joinpath(f"data/{inputs.experiment_name}").mkdir(
        parents=True, exist_ok=True
    )
    scores.to_csv(
        subdirectory.joinpath(
            f"data/{inputs.experiment_name}/"
            f"{analysis_obj.analysis_details.loadings}_scores.csv"
        )
    )
