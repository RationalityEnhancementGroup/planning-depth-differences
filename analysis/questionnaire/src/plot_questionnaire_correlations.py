from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from costometer.utils import set_font_sizes


def plot_questionnaire_pairs(numeric_combined_scores, scale=8):
    set_font_sizes(SMALL_SIZE=8 * scale, MEDIUM_SIZE=10 * scale, BIGGER_SIZE=15 * scale)

    plt.figure(figsize=(12 * scale, 8 * scale), dpi=80)

    corr_df = numeric_combined_scores.corr()

    sns.heatmap(
        corr_df, annot=True, fmt=".2f", annot_kws={"size": 6 * scale}, cmap="vlag"
    )


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

    data_path.joinpath("figs").mkdir(parents=True, exist_ok=True)
    plt.savefig(
        data_path.joinpath(f"figs/{inputs.experiment_name}_questionnaire_corr.png"),
        bbox_inches="tight",
    )
