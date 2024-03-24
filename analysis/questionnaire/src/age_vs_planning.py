import logging
import sys
from pathlib import Path

import dill as pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from costometer.utils import (
    get_correlation_text,
    get_mann_whitney_text,
    standard_parse_args,
)
from mouselab.cost_functions import forward_search_cost
from mouselab.mouselab import MouselabEnv
from scipy.stats import mode

"""
TODO: hack to finish things quicker, would be better to move the whole
calculation of forward search trials out
"""
sys.path.append(str(Path(__file__).parents[3].joinpath("cluster")))
from src.cluster_utils import get_human_trajectories  # noqa: E402

if __name__ == "__main__":
    irl_path = Path(__file__).resolve().parents[3]
    analysis_obj, inputs, subdirectory = standard_parse_args(
        description=sys.modules[__name__].__doc__,
        irl_path=irl_path,
        filename=Path(__file__).stem,
        default_experiment="QuestMain",
        default_subdirectory="questionnaire",
    )

    optimization_data = analysis_obj.query_optimization_data(
        excluded_parameters=analysis_obj.analysis_details.excluded_parameters
    )

    model_parameters = list(
        set(analysis_obj.cost_details.constant_values)
        - set(analysis_obj.analysis_details.excluded_parameters)
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
            f"{analysis_obj.analysis_details.loadings}_scores.csv"
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

    logging.info("-------")
    for model_parameter in model_parameters:
        logging.info("Model parameter: {model_parameter}")
        logging.info(
            get_correlation_text(
                pg.corr(combined_scores["age"], combined_scores[model_parameter])
            )
        )
    logging.info("-------")

    # now we check which CM are correlated with age
    cm = {}
    for session in analysis_obj.analysis_details.sessions:
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
        mwu_obj = pg.mwu(
            combined_scores[combined_scores[strategy].astype(bool)]["age"].values,
            combined_scores[~combined_scores[strategy].astype(bool)]["age"].values,
        )

        if mwu_obj["p-val"][0] < 0.05:
            logging.info(strategy)
            logging.info(get_mann_whitney_text(mwu_obj))
            logging.info(
                f"M (strategy): "
                f"{combined_scores[combined_scores[strategy].astype(bool)]['age'].mean():.03f}"  # noqa : E501
                f", M (not strategy): "
                f"{combined_scores[~combined_scores[strategy].astype(bool)]['age'].mean():.03f}"  # noqa : E501
            )

    logging.info("-------")

    logging.info("Forward strategy cluster")
    combined_scores["forward_strategy"] = combined_scores.apply(
        lambda row: sum(
            [
                row[f"cm_{strategy}"]
                for strategy in [3, 10, 82, 5, 36, 37, 54, 79]
                if f"cm_{strategy}" in row
            ]
        ),
        axis=1,
    )

    mwu_obj = pg.mwu(
        combined_scores[combined_scores["forward_strategy"].astype(bool)]["age"].values,
        combined_scores[~combined_scores["forward_strategy"].astype(bool)][
            "age"
        ].values,
        alternative="greater",
    )
    logging.info(get_mann_whitney_text(mwu_obj))
    logging.info(
        f"M (strategy): "
        f"{combined_scores[combined_scores['forward_strategy'].astype(bool)]['age'].mean():.03f}"  # noqa : E501
        f", M (not strategy): "
        f"{combined_scores[~combined_scores['forward_strategy'].astype(bool)]['age'].mean():.03f}"  # noqa : E501
    )
    logging.info("-------")
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

    logging.info(
        "Sanity check -- forward trials should be correlated with forward bonus"
    )
    logging.info(
        get_correlation_text(
            pg.corr(
                combined_scores["forw_added_cost"], combined_scores["forward_trials"]
            )
        )
    )

    logging.info("Correlation between age and structure understanding")
    logging.info(
        get_correlation_text(
            pg.corr(combined_scores["age"], combined_scores["structure_understanding"])
        )
    )
    logging.info("Correlation between age and forward trials")
    logging.info(
        get_correlation_text(
            pg.corr(combined_scores["age"], combined_scores["forward_trials"])
        )
    )
    logging.info("Correlation between structure understanding and forward trials")
    logging.info(
        get_correlation_text(
            pg.corr(
                combined_scores["structure_understanding"],
                combined_scores["forward_trials"],
            )
        )
    )

    logging.info("-------")
    for model_parameter in model_parameters:
        logging.info("Model parameter: {model_parameter}")
        logging.info(
            get_correlation_text(
                pg.corr(
                    combined_scores[combined_scores["forward_trials"] != 20]["age"],
                    combined_scores[combined_scores["forward_trials"] != 20][
                        model_parameter
                    ],
                )
            )
        )
    logging.info("-------")

    sns.distplot(combined_scores["forward_trials"])
    plt.savefig(
        subdirectory.joinpath(f"figs/{inputs.experiment_name}_forward_trials.png"),
        bbox_inches="tight",
    )

    with open(
        subdirectory.joinpath(f"data/{inputs.experiment_name}_forward_trials.pkl"), "wb"
    ) as f:
        pickle.dump(forward_planning_trial, f)
