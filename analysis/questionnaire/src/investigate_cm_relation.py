import logging
import sys
from collections import Counter
from pathlib import Path

import dill as pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
import yaml
from costometer.utils import get_mann_whitney_text, standard_parse_args
from scipy.stats import mode


def plot_cm_relation(numeric_combined_scores, strategy, col):
    plt.figure(figsize=(12, 8), dpi=80)

    numeric_combined_scores[f"Strategy {strategy}"] = numeric_combined_scores.apply(
        lambda row: row["strategy"] == strategy, axis=1
    )
    sns.violinplot(x=f"Strategy {strategy}", y=col, data=numeric_combined_scores)
    plt.title(f"Strategy {strategy}, metric {col}")
    del numeric_combined_scores[f"Strategy {strategy}"]


if __name__ == "__main__":
    irl_path = Path(__file__).resolve().parents[3]
    analysis_obj, inputs, subdirectory = standard_parse_args(
        description=sys.modules[__name__].__doc__,
        irl_path=irl_path,
        filename=Path(__file__).stem,
        default_experiment="QuestMain",
        default_subdirectory="questionnaire",
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

    psych_scores = pd.read_csv(
        irl_path.joinpath(
            f"analysis/questionnaire/data/{inputs.experiment_name}/"
            f"{analysis_obj.analysis_details.loadings}_scores.csv"
        )
    )
    numeric_combined_scores = numeric_combined_scores.merge(psych_scores, on="pid")

    participant_strategies = {}
    for session in experiment_arguments["sessions"]:
        with open(irl_path.joinpath(f"cluster/data/cm/{session}.pkl"), "rb") as f:
            exp = pickle.load(f)
            participant_strategies = {
                **exp.participant_strategies,
                **participant_strategies,
            }

    # IMPROVE: refactor this, sort of duplicated in find_stable_point_with_cm.py
    mode_vals = {}
    for pid, strategies in exp.participant_strategies.items():
        last_strategies = strategies[-10:]
        mode_vals[pid] = mode(last_strategies).mode

    numeric_combined_scores["optimal"] = numeric_combined_scores["pid"].apply(
        lambda pid: mode_vals[pid] == 21
    )
    numeric_combined_scores["strategy"] = numeric_combined_scores["pid"].apply(
        lambda pid: mode_vals[pid]
    )

    numeric_combined_scores.groupby(["optimal"]).mean()

    counter = Counter(mode_vals.values()).most_common()
    minimum_count = round(0.05 * (len(mode_vals.values())))

    most_common = [
        strategy
        for strategy, strategy_count in counter
        if strategy_count >= minimum_count
    ]

    for strategy in most_common:
        for col in list(numeric_combined_scores):
            if (
                len(
                    np.unique(
                        numeric_combined_scores[
                            numeric_combined_scores["strategy"] == strategy
                        ][col].values
                    )
                )
                > 1
                and len(
                    np.unique(
                        numeric_combined_scores[
                            numeric_combined_scores["strategy"] != strategy
                        ][col].values
                    )
                )
                > 1
            ):
                mwu_obj = pg.mwu(
                    numeric_combined_scores[
                        numeric_combined_scores["strategy"] == strategy
                    ][col].values,
                    numeric_combined_scores[
                        numeric_combined_scores["strategy"] != strategy
                    ][col].values,
                )
                if mwu_obj["p-val"][0] < 0.05:
                    logging.info(strategy, col)
                    logging.info(get_mann_whitney_text(mwu_obj))

                    logging.info(
                        "%.3f %.3f",
                        numeric_combined_scores[
                            numeric_combined_scores["strategy"] == strategy
                        ][
                            col
                        ].mean(),  # noqa : E501
                        numeric_combined_scores[
                            numeric_combined_scores["strategy"] != strategy
                        ][
                            col
                        ].mean(),  # noqa : E501
                    )

                    plot_cm_relation(numeric_combined_scores, strategy, col)
                    plt.savefig(
                        subdirectory.joinpath(
                            f"figs/{inputs.experiment_name}_CM_{strategy}_{col}.png"
                        ),
                        bbox_inches="tight",
                    )
