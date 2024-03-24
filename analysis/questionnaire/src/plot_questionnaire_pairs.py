from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from costometer.utils import set_font_sizes, set_plotting_and_logging_defaults


def plot_questionnaire_pairs(numeric_combined_scores, scale=5):
    set_font_sizes(SMALL_SIZE=8 * scale, MEDIUM_SIZE=10 * scale, BIGGER_SIZE=15 * scale)

    plt.figure(figsize=(12 * scale, 8 * scale), dpi=80)

    # https://stackoverflow.com/a/64916160
    not_na = numeric_combined_scores.notna().astype(int)

    sns.heatmap(
        not_na.T.dot(not_na),
        annot=True,
        fmt="d",
        annot_kws={"size": 8 * scale},
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp",
        dest="experiment_name",
    )
    inputs = parser.parse_args()

    subdirectory = Path(__file__).resolve().parents[1]
    irl_path = Path(__file__).resolve().parents[3]

    set_plotting_and_logging_defaults(
        subdirectory=subdirectory,
        experiment_name="QuestionnairePairs",
        filename=Path(__file__).stem,
    )

    with open(
        subdirectory.joinpath(f"inputs/yamls/{inputs.experiment_name}.yaml"), "r"
    ) as stream:
        experiment_arguments = yaml.safe_load(stream)

    combined_scores = pd.concat(
        [
            pd.read_csv(
                irl_path.joinpath(f"data/processed/{session}/combined_scores.csv")
            )
            for session in experiment_arguments["sessions"]
        ]
    )
    numeric_combined_scores = combined_scores[
        combined_scores.columns.difference(["gender"])
    ]

    plot_questionnaire_pairs(numeric_combined_scores)

    subdirectory.joinpath("figs").mkdir(parents=True, exist_ok=True)
    plt.savefig(
        subdirectory.joinpath(f"figs/{inputs.experiment_name}_questionnaire_pairs.png"),
        bbox_inches="tight",
    )
