"""From the mcl_toolbox, maintained by the REG."""
import numpy as np
from scipy.special import logsumexp, softmax

from ..env.modified_mouselab import TrialSequence
from .learning_utils import get_normalized_feature_values


def get_accuracy_position(  # noqa : N806
    position, ground_truth, clicks, pipeline, features, normalized_features, W
):
    num_features = len(features)
    env = TrialSequence(1, pipeline, ground_truth=[ground_truth])
    trial = env.trial_sequence[0]
    beta = 1
    acc = []
    total_neg_click_likelihood = 0
    for click in clicks:
        unobserved_nodes = trial.get_unobserved_nodes()
        unobserved_node_labels = [node.label for node in unobserved_nodes]
        click_index = unobserved_node_labels.index(click)
        feature_values = np.zeros((len(unobserved_nodes), num_features))
        for i, node in enumerate(unobserved_nodes):
            feature_values[i] = node.compute_termination_feature_values(features)
            if normalized_features:
                feature_values[i] = get_normalized_feature_values(
                    feature_values[i], features, normalized_features
                )
        dot_product = beta * np.dot(W, feature_values.T)
        softmax_dot = softmax(dot_product)
        neg_log_likelihood = -np.log(softmax_dot[click_index])
        total_neg_click_likelihood += neg_log_likelihood
        sorted_indices = np.argsort(dot_product)[::-1]
        sorted_list_indices = get_sorted_list_indices(sorted_indices, dot_product)
        sorted_list_clicks = [
            [unobserved_node_labels[index] for index in indices]
            for indices in sorted_list_indices
        ]
        click_present = False
        for clicks_list in sorted_list_clicks[:position]:
            if click in clicks_list:
                click_present = True
                break
        if click_present:
            acc.append(1)
        else:
            acc.append(0)
        trial.node_map[click].observe()
    average_click_likelihood = np.exp((-1 / len(clicks)) * total_neg_click_likelihood)
    return acc, average_click_likelihood


def get_acls(
    strategies,
    pids,
    p_envs,
    p_clicks,
    pipeline,
    features,
    normalized_features,
    strategy_weights,
):
    acls = []
    random_acls = []
    total_acc = []
    for pid in pids:
        if pid in p_envs:
            # print(pid)
            envs = p_envs[pid]
            clicks = p_clicks[pid]
            pid_acc = []
            for i in range(len(envs)):
                strategy_num = strategies[pid][i]
                pid_acc, acl = get_accuracy_position(
                    1,
                    envs[i],
                    clicks[i],
                    pipeline,
                    features,
                    normalized_features,
                    strategy_weights[strategy_num - 1],
                )
                _, random_acl = get_accuracy_position(
                    1,
                    envs[i],
                    clicks[i],
                    pipeline,
                    features,
                    normalized_features,
                    strategy_weights[38],
                )
                acls.append(acl)
                random_acls.append(random_acl)
            total_acc += pid_acc
    # print(np.sum(total_acc)/len(total_acc))
    return acls, random_acls


def get_sorted_list_indices(sorted_indices, dot_product):
    total_list = []
    temp_list = [sorted_indices[0]]
    for index in sorted_indices[1:]:
        dp = dot_product[index]
        if not temp_list or dp == dot_product[temp_list[-1]]:
            temp_list.append(index)
        else:
            total_list.append(temp_list)
            temp_list = []
    return total_list


def compute_trial_feature_log_likelihood(
    trial, trial_features, click_sequence, weights, inv_t=False
):
    trial.reset_observations()
    log_likelihoods = []
    feature_len = weights.shape[0]
    beta = 1
    W = weights  # noqa : N806
    if inv_t:
        feature_len -= 1
        beta = weights[-1]
        W = weights[:-1]  # noqa : N806
    for i in range(len(click_sequence)):
        click = click_sequence[i]
        unobserved_nodes = trial.get_unobserved_nodes()
        unobserved_node_labels = [node.label for node in unobserved_nodes]
        feature_values = trial_features[i][unobserved_node_labels, :]
        dot_product = beta * np.dot(W, feature_values.T)
        click_index = unobserved_node_labels.index(click)
        trial.node_map[click].observe()
        log_lik = dot_product[click_index] - logsumexp(dot_product)
        log_likelihoods.append(log_lik)
    return np.sum(log_likelihoods)


def compute_current_features(trial, features, normalized_features):
    num_nodes = trial.num_nodes
    num_features = len(features)
    action_feature_values = np.zeros((num_nodes, num_features))
    for node_num in range(num_nodes):
        node = trial.node_map[node_num]
        action_feature_values[node_num] = node.compute_termination_feature_values(
            features
        )
        if normalized_features:
            action_feature_values[node_num] = get_normalized_feature_values(
                action_feature_values[node_num], features, normalized_features
            )
    return action_feature_values


def compute_trial_features(
    pipeline, ground_truth, trial_actions, features_list, normalized_features
):
    num_features = len(features_list)
    env = TrialSequence(num_trials=1, pipeline=pipeline, ground_truth=[ground_truth])
    trial = env.trial_sequence[0]
    num_actions = len(trial_actions)
    num_nodes = trial.num_nodes
    action_feature_values = np.zeros((num_actions, num_nodes, num_features))
    for i, action in enumerate(trial_actions):
        node_map = trial.node_map
        trial_feature_values = compute_current_features(
            trial, features_list, normalized_features
        )
        for node_num in range(num_nodes):
            action_feature_values[i][node_num] = trial_feature_values[node_num]
        node_map[action].observe()
    return action_feature_values
