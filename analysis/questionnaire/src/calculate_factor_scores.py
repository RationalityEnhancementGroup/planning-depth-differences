from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import yaml
from quest_utils.factor_utils import get_psychiatric_scores, load_weights

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp",
        dest="experiment_name",
    )
    inputs = parser.parse_args()

    data_path = Path(__file__).resolve().parents[1]
    irl_path = Path(__file__).resolve().parents[3]

    with open(
        data_path.joinpath(f"inputs/yamls/{inputs.experiment_name}.yaml"), "r"
    ) as stream:
        experiment_arguments = yaml.safe_load(stream)

    weights = load_weights(
        data_path.joinpath(f"inputs/loadings/{experiment_arguments['loadings']}.csv"),
        wise_weights=True,
    )

    questionnaires = pd.concat(
        [
            pd.read_csv(
                irl_path.joinpath(f"data/processed/{session}/questionnaires.csv")
            )
            for session in experiment_arguments["sessions"]
        ]
    )

    # TODO add this to preprocessing
    questionnaires["question_id"] = questionnaires.apply(
        lambda row: "catch.1"
        if (row["name"] == "UPPS-P") & (row["question_id"] == "catch.2")
        else row["question_id"],
        axis=1,
    )
    # TODO have as numeric originally
    questionnaires = questionnaires[
        questionnaires["score"].apply(lambda score: str(score).isnumeric())
    ]
    questionnaires["score"] = questionnaires["score"].apply(int)

    individual_items = questionnaires.pivot_table(
        index=["pid", "run"], columns="question_id", values="score"
    )

    scores = get_psychiatric_scores(individual_items, weights, scale_cols=True)

    data_path.joinpath(f"data/{inputs.experiment_name}").mkdir(
        parents=True, exist_ok=True
    )
    scores.to_csv(data_path.joinpath(f"data/{inputs.experiment_name}/scores.csv"))
