from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

import dill as pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from costometer.utils import AnalysisObject, get_ttest_text, set_font_sizes
from scipy.stats import mode

set_font_sizes()

if __name__ == "__main__":
    """
    Example:
    python src/find_stable_point_with_cm.py -e MainExperiment
    """
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp",
        dest="experiment_name",
        metavar="experiment_name",
    )
    parser.add_argument(
        "-s",
        "--subdirectory",
        dest="experiment_subdirectory",
        metavar="experiment_subdirectory",
    )
    inputs = parser.parse_args()

    irl_path = Path(__file__).resolve().parents[4]
    data_path = irl_path.joinpath(f"analysis/{inputs.experiment_subdirectory}")

    analysis_obj = AnalysisObject(
        inputs.experiment_name,
        irl_path=irl_path,
        experiment_subdirectory=inputs.experiment_subdirectory,
    )

    assert (
        len(
            np.unique(
                [
                    session_details["experiment_setting"]
                    for session_details in analysis_obj.session_details.values()
                ]
            )
        )
        == 1
    )
    experiment_setting = np.unique(
        [
            session_details["experiment_setting"]
            for session_details in analysis_obj.session_details.values()
        ]
    )[0]

    participant_strategies = {}
    for session in analysis_obj.session_details:
        with open(irl_path.joinpath(f"cluster/data/cm/{session}.pkl"), "rb") as f:
            exp = pickle.load(f)
            participant_strategies = {
                **exp.participant_strategies,
                **participant_strategies,
            }

    mode_vals = []
    mode_counts = []
    strategy_counts = []
    for pid, strategies in exp.participant_strategies.items():

        last_strategies = strategies[-20:]

        mode_vals.append(["last", mode(last_strategies).mode])
        mode_counts.append(["last", mode(last_strategies).count[0]])
        mode_counts.append(["full", mode(strategies).count[0]])

        strategy_counts.append(["last", len(np.unique(last_strategies))])
        strategy_counts.append(["full", len(np.unique(strategies))])
    mode_count_df = pd.DataFrame(mode_counts, columns=["type", "count"])
    strategy_count_df = pd.DataFrame(strategy_counts, columns=["type", "count"])

    plt.figure(figsize=(11.7, 8.27))
    sns.violinplot(x="type", y="count", data=mode_count_df)
    plt.title("Number of trials using mode strategy in trial subset")
    plt.savefig(
        data_path.joinpath(f"figs/{inputs.experiment_name}_stable_point_mode.png"),
        bbox_inches="tight",
    )

    plt.figure(figsize=(11.7, 8.27))
    sns.violinplot(x="type", y="count", data=strategy_count_df)
    plt.title("Number of strategies in trial subset")
    plt.savefig(
        data_path.joinpath(f"figs/{inputs.experiment_name}_stable_point_count.png"),
        bbox_inches="tight",
    )

    print("Number of strategies per participant in trial subset (1 is better)")
    print(
        strategy_count_df[strategy_count_df["type"] == "last"]
        .groupby(["count"])
        .count()
    )

    print("Into percentags for reporting")
    print(
        strategy_count_df[strategy_count_df["type"] == "last"]
        .groupby(["count"])
        .count()
        / len(strategy_count_df[strategy_count_df["type"] == "last"])
    )

    print(
        "Number of trials using mode strategy per participant "
        "in trial subset (20 is better)"
    )
    print(mode_count_df[strategy_count_df["type"] == "last"].groupby(["count"]).count())

    print("T-Test 1 vs number of strategies per participant")

    ttest_obj = pg.ttest(
        x=strategy_count_df[strategy_count_df["type"] == "last"]["count"].values, y=1
    )
    print(get_ttest_text(ttest_obj))
    print(
        f"$M: {strategy_count_df[strategy_count_df['type'] == 'last']['count'].mean():.3f}, "  # noqa: E501
        f"SD: {strategy_count_df[strategy_count_df['type'] == 'last']['count'].std():.3f}$"  # noqa: E501
    )

    print("Most common strategies (strategy number, number of participants)")
    print(Counter([val[1][0] for val in mode_vals]).most_common())
