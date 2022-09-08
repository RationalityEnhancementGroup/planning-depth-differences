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

    individual_items = pd.concat(
        [
            pd.read_csv(
                irl_path.joinpath(f"data/processed/{session}/individual_items.csv")
            )
            for session in experiment_arguments["sessions"]
        ]
    )
    individual_items = individual_items[individual_items.columns.difference(["gender"])]
    individual_items = individual_items.set_index("pid")

    scores = get_psychiatric_scores(individual_items, weights, scale_cols=True)

    scores.to_csv(data_path.joinpath(f"data/{inputs.experiment_name}/scores.csv"))
