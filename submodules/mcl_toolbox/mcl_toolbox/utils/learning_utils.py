"""From the mcl_toolbox, maintained by the REG."""
import pickle
from collections import Counter, defaultdict
from functools import partial

import numpy as np
from mouselab.analysis_utils import get_data
from mouselab.distributions import Categorical, Normal
from mouselab.envs.registry import registry


def pickle_load(file_path):
    """
    Load the pickle file located at 'filepath'
    Params:
        file_path  -- Location of the file to be loaded, as pathlib object.
    Returns:
        Unpickled object
    """
    with open(str(file_path), "rb") as file_obj:
        unpickled_obj = pickle.load(file_obj)
    return unpickled_obj


def get_clicks(exp_num="v1.0", data_path=None):
    """
    Get clicks made by a particular participant
    Params:
        exp_num : Experiment number according to the experiment folder
        participant_num : The id of the participants to get the environments for
    Returns:
        clicks_data: The clicks made by the participant in all trials.
    """
    data = get_data(exp_num, data_path)
    mdf = data["mouselab-mdp"]
    clicks_data = defaultdict(list)
    for _, row in mdf.iterrows():
        pid = row.pid
        queries = row.queries["click"]["state"]["target"]
        queries = [int(query) for query in queries]
        queries.append(0)
        clicks_data[pid].append(queries)
    clicks_data = dict(clicks_data)
    return clicks_data


def get_participant_scores(exp_num="v1.0", num_participants=166, data_path=None):
    """
    Get scores of participants
    Params:
        exp_num : Experiment number according to the experiment folder.
        num_participants: Max pid+1 to consider
    Returns:
        A dictionary of scores of participants with pid as key and rewards as values.
    """
    data = get_data(exp_num, data_path)
    mdf = data["mouselab-mdp"]
    participant_scores = {}
    # for participant_num in range(num_participants):
    for (
        participant_num
    ) in num_participants:  # changed this to output score for a set list of pid's
        score_list = list(mdf[mdf.pid == participant_num]["score"])
        participant_scores[participant_num] = score_list
    return participant_scores


def construct_repeated_pipeline(branching, reward_function, num_trials):
    return [(branching, reward_function)] * num_trials


def create_mcrl_reward_distribution(experiment_setting, reward_dist="categorical"):
    """
    Creates MCRL reward distribution given experiment setting ala mouselab package
    @param experiment_setting: on experiment registry in mouselab package
    @param reward_dist: type of probability distribution (e.g. Categorical, Normal)
    @return: reward_distribution as the MCRL project uses
    """
    reward_dictionary = registry(experiment_setting).reward_dictionary
    reward_inputs = [
        reward_dictionary[depth + 1].vals for depth in range(len(reward_dictionary))
    ]

    # build reward distribution for pipeline with experiment details
    reward_distributions = construct_reward_function(
        reward_inputs,
        reward_dist,
    )
    return reward_distributions


def reward_function(depth, level_distributions):
    if depth > 0:
        return level_distributions[depth - 1]
    return 0.0


def combine_level_dists(level_distributions):
    func = partial(reward_function, level_distributions=level_distributions)
    return func


def construct_reward_function(params_list, dist_type="categorical"):
    if dist_type.lower() == "categorical":
        level_distributions = [Categorical(param) for param in params_list]
    elif dist_type.lower() == "normal":
        level_distributions = [Normal(param[0], param[1]) for param in params_list]
    else:
        raise ValueError("Please select one of categorical or normal distributions")
    return combine_level_dists(level_distributions)


def sidak_value(significance_threshold, num_tests):
    return 1 - (1 - significance_threshold) ** (1 / num_tests)


def get_normalized_feature_values(feature_values, features_list, max_min_values):
    """
    Get the normalized feature values
    """
    normalized_features = np.array(feature_values)
    if max_min_values:
        max_feature_values, min_feature_values = max_min_values
        for i, (feature, fv) in enumerate(zip(features_list, feature_values)):
            max_min_diff = max_feature_values[feature] - min_feature_values[feature]
            f_min_diff = fv - min_feature_values[feature]
            # print(feature, f_min_diff, max_min_diff, max_min_diff - f_min_diff)
            if feature == "constant":
                normalized_features[i] = 1
            elif max_min_diff == 0:
                normalized_features[i] = 0
            else:
                normalized_features[i] = f_min_diff / max_min_diff
    return normalized_features


def get_counts(strategies, num_trials):
    new_strategies_list = list(strategies.values())
    new_strategies_list = [S for S in new_strategies_list if len(S) == num_trials]
    strategies_data = np.array(new_strategies_list)
    strategies_data = strategies_data.flatten()
    counts = Counter(strategies_data)
    ns = strategies_data.shape[0]
    counts = {k: v / ns for k, v in counts.items()}
    return counts


def get_modified_weights(strategy_space, weights):
    num_strategies = len(strategy_space)
    W = np.zeros((num_strategies, weights.shape[1]))
    for i, s in enumerate(strategy_space):
        W[i] = weights[s - 1]
    return W
