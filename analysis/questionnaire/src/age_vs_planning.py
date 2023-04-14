from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import mode
import dill as pickle
from costometer.utils import AnalysisObject
import pingouin as pg
import numpy as np
import pandas as pd
import sys
from mouselab.cost_functions import forward_search_cost
from mouselab.mouselab import MouselabEnv

from costometer.utils import get_correlation_text

"""
TODO: hack to finish things quicker, would be better to move the whole
calculation of forward search trials out
"""
sys.path.append(str(Path(__file__).parents[3].joinpath("cluster")))
from src.cluster_utils import get_human_trajectories  # noqa: E402

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp",
        default="QuestMain",
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
    optimization_data = analysis_obj.query_optimization_data(
        excluded_parameters=analysis_obj.excluded_parameters
    )

    model_parameters = list(
        set(analysis_obj.cost_details["constant_values"])
        - set(analysis_obj.excluded_parameters.split(","))
    )

    combined_scores = analysis_obj.dfs["combined_scores"].copy(deep=True)

    combined_scores = combined_scores.reset_index().merge(
        optimization_data[model_parameters + ["trace_pid"]],
        left_on="pid",
        right_on="trace_pid",
    )

    # add psychiatric factor scores
    factor_scores = pd.read_csv(
        irl_path.joinpath(
            f"analysis/questionnaire/data/{inputs.experiment_name}/"
            f"{analysis_obj.loadings}_scores.csv"
        )
    )
    combined_scores = combined_scores.merge(factor_scores)

    analysis_obj.dfs["quiz-and-demo"]["structure_understanding"] = analysis_obj.dfs[
        "quiz-and-demo"
    ].apply(
        lambda row: np.sum(
            [row[f"mouselab-quiz-post_Q{qidx}"] for qidx in range(1, 4)]
        ),
        axis=1,
    )
    combined_scores = combined_scores.merge(
        analysis_obj.dfs["quiz-and-demo"][
            ["pid", "age", "gender", "structure_understanding"]
        ]
    )

    print("-------")
    for model_parameter in model_parameters:
        print(f"Model parameter: {model_parameter}")
        print(
            get_correlation_text(
                pg.corr(combined_scores["age"], combined_scores[model_parameter])
            )
        )
    print("-------")

    # now we check which CM are correlated with age
    cm = {}
    for session in analysis_obj.sessions:
        with open(irl_path.joinpath(f"cluster/data/cm/{session}.pkl"), "rb") as f:
            exp = pickle.load(f)
        for pid, strategies in exp.participant_strategies.items():
            last_strategies = strategies[-20:]
            cm[pid] = mode(last_strategies).mode[0]

    combined_scores["mode_strategy"] = combined_scores["pid"].apply(
        lambda curr_pid: cm[curr_pid]
    )
    dummies = pd.get_dummies(combined_scores["mode_strategy"], prefix="cm")
    combined_scores = pd.concat([combined_scores, dummies], axis=1)

    for strategy in dummies.columns.values:
        corr_obj = pg.corr(combined_scores["age"], combined_scores[strategy])
        if corr_obj["p-val"][0] < 0.05:
            print(strategy)
            print(get_correlation_text(corr_obj))

    print("-------")

    # calculate forward search trials
    traces = get_human_trajectories(
        "quest_main",
        pids=None,
        blocks=["test"],
        include_last_action=False,
    )

    cost = forward_search_cost(added_cost=-1, inspection_cost=0)
    env = MouselabEnv.new_symmetric_registered("high_increasing", cost=cost)

    forward_planning_trial = {}
    for trace in traces:
        forward_planning_trial[trace["pid"][0]] = 0
        for trial_actions in trace["actions"]:
            env.reset()
            trial_sum = 0
            for action in trial_actions:
                if action != 13:
                    _, r, _, _ = env.step(action)
                    trial_sum += r
            if len(trial_actions) > 1:
                forward_planning_trial[trace["pid"][0]] += (
                    len(
                        [
                            action
                            for action in trial_actions
                            if action not in [1, 5, 9, 13]
                        ]
                    )
                    == trial_sum
                )

    combined_scores["forward_trials"] = combined_scores["pid"].apply(
        lambda curr_pid: forward_planning_trial[curr_pid]
    )

    print("Sanity check -- forward trials should be correlated with forward bonus")
    print(
        get_correlation_text(
            pg.corr(
                combined_scores["forw_added_cost" ""], combined_scores["forward_trials"]
            )
        )
    )

    print("Correlation between age and structure understanding")
    print(
        get_correlation_text(
            pg.corr(combined_scores["age"], combined_scores["structure_understanding"])
        )
    )
    print("Correlation between age and forward trials")
    print(
        get_correlation_text(
            pg.corr(combined_scores["age"], combined_scores["forward_trials"])
        )
    )
    print("Correlation between structure understanding and forward trials")
    print(
        get_correlation_text(
            pg.corr(
                combined_scores["structure_understanding"],
                combined_scores["forward_trials"],
            )
        )
    )

    print("-------")
    for model_parameter in model_parameters:
        print(f"Model parameter: {model_parameter}")
        print(
            get_correlation_text(
                pg.corr(
                    combined_scores[combined_scores["forward_trials"] != 20]["age"],
                    combined_scores[combined_scores["forward_trials"] != 20][
                        model_parameter
                    ],
                )
            )
        )
    print("-------")

    sns.distplot(combined_scores["forward_trials"])
    plt.savefig(
        data_path.joinpath(f"figs/{inputs.experiment_name}_forward_trials.png"),
        bbox_inches="tight",
    )

    with open(
        data_path.joinpath(f"data/{inputs.experiment_name}_forward_trials.pkl"), "wb"
    ) as f:
        pickle.dump(forward_planning_trial, f)
