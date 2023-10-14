"""From the mcl_toolbox, maintained by the REG."""
from pathlib import Path

import numpy as np
from mouselab.envs.registry import registry

from ..env.modified_mouselab import TrialSequence
from .learning_utils import (
    construct_repeated_pipeline,
    create_mcrl_reward_distribution,
    pickle_load,
)
from .planning_strategies import strategy_dict
from .sequence_utils import compute_trial_features

strategy_space = pickle_load(Path(__file__).parents[0] / "data" / "strategy_space.pkl")


def generate_data(strategy_num, pipeline, num_simulations=100, rng=None):
    rng = np.random.default_rng(rng)
    envs = [TrialSequence(len(pipeline), pipeline) for _ in range(num_simulations)]
    ground_truths = []
    simulated_actions = []
    for env in envs:
        ground_truths.extend(env.ground_truth)
        for trial in env.trial_sequence:
            actions = strategy_dict[strategy_num](trial, rng=rng)
            simulated_actions.append(actions)
    return ground_truths, simulated_actions


def normalize(pipeline, features_list, num_simulations=100, rng=None):
    rng = np.random.default_rng(rng)
    simulated_features = []
    for strategy_num in strategy_space:
        ground_truths, simulated_actions = generate_data(
            strategy_num, pipeline, num_simulations=num_simulations, rng=rng
        )
        for curr_ground_truth, curr_simulated_actions in zip(
            ground_truths, simulated_actions
        ):
            trial_features = compute_trial_features(
                pipeline,
                curr_ground_truth,
                curr_simulated_actions,
                features_list,
                False,
            )
            simulated_features += trial_features.tolist()
    simulated_features = np.array(simulated_features)
    simulated_features_shape = simulated_features.shape
    simulated_features = simulated_features.reshape(-1, simulated_features_shape[-1])
    max_feature_values = np.max(simulated_features, axis=0)
    min_feature_values = np.min(simulated_features, axis=0)
    max_feature_values = {f: max_feature_values[i] for i, f in enumerate(features_list)}
    min_feature_values = {f: min_feature_values[i] for i, f in enumerate(features_list)}
    return max_feature_values, min_feature_values


def get_new_feature_normalization(
    features_list,
    exp_setting="high_increasing",
    num_trials=30,
    num_simulations=100,
    rng=None,
):
    rng = np.random.default_rng(rng)
    branching = registry(exp_setting).branching
    reward_distributions = create_mcrl_reward_distribution(exp_setting)
    pipeline = construct_repeated_pipeline(branching, reward_distributions, num_trials)
    max_fv, min_fv = normalize(
        pipeline, features_list, num_simulations=num_simulations, rng=rng
    )
    return max_fv, min_fv
