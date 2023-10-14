"""From the mcl_toolbox, maintained by the REG."""
from collections import defaultdict

import numpy as np
from hyperopt import fmin, hp, tpe
from hyperopt.fmin import generate_trials_to_calculate

from ..env.modified_mouselab import TrialSequence
from ..utils.sequence_utils import (
    compute_trial_feature_log_likelihood,
    compute_trial_features,
)


class ComputationalMicroscope:
    """
    The computational Microscope finds out the underlying strategies from the click sequences.
    Params:
        transition_T: The temperature parameter for the transition matrix based on distances
    """

    def __init__(
        self,
        pipeline,
        strategy_space,
        strategy_weights,
        features,
        normalized_features=True,
    ):
        self.pipeline = pipeline
        self.strategy_space = strategy_space
        self.strategy_weights = strategy_weights
        self.num_strategies = len(self.strategy_space)
        if strategy_weights.shape[0] != self.num_strategies:
            weight_error = "Please check the weights passed. Only pass weights for the strategies in the strategy space."
            raise ValueError(weight_error)
        self.features = features
        self.normalized_features = normalized_features
        self.strategy_T = 1
        self.T_grad = (
            np.ones((self.num_strategies, self.num_strategies)) / self.num_strategies
        )

    def set_strategy_T(self, strategy_T):
        self.strategy_T = strategy_T

    def get_trial_log_likelihood(
        self, trial, trial_features, click_sequence, strategy_index
    ):
        weights = np.append(self.strategy_weights[strategy_index], 1 / self.strategy_T)
        log_likelihood = compute_trial_feature_log_likelihood(
            trial, trial_features, click_sequence, weights, inv_t=True
        )
        return log_likelihood

    def compute_trials_likelihood(self, click_sequences, envs):
        num_trials = len(click_sequences)
        log_likelihoods = []
        trials = TrialSequence(num_trials, self.pipeline, envs)
        for trial_num in range(num_trials):
            trial = trials.trial_sequence[trial_num]
            click_sequence = click_sequences[trial_num]
            trial_features = compute_trial_features(
                self.pipeline,
                envs[trial_num],
                click_sequence,
                self.features,
                self.normalized_features,
            )
            log_likelihoods.append(
                [
                    self.get_trial_log_likelihood(
                        trial, trial_features, click_sequence, i
                    )
                    for i in range(self.num_strategies)
                ]
            )
        return log_likelihoods

    def viterbi(self, T, L, prior=None):
        """Returns MLE state sequence and likelihood.

        Arguments:
            L: Log-likelihood matrix such that L[i,j] = log(p(x_i | z_j))
            T: Transition matrix such that T[i, j] is the log-probability of
               transitioning from state z_i to state z_j
        """
        assert L.max() <= 0
        assert T.max() <= 0
        n_obs, n_state = L.shape
        prior = np.zeros(n_state) - np.log(n_state) if prior is None else prior
        prob = np.zeros((n_obs, n_state))
        prev = np.zeros((n_obs, n_state))
        prob[0] = prior + L[0]
        for i in range(1, n_obs):
            for j in range(n_state):
                pk = prob[i - 1, :] + T[:, j] + L[i, j]
                k = np.argmax(pk)
                prob[i, j] = pk[k]
                prev[i, j] = k

        x = np.zeros(n_obs, dtype="int")
        x[-1] = np.argmax(prob[-1])
        for i in range(n_obs - 1, 0, -1):
            x[i - 1] = prev[i, x[i]]
        return x, prob[-1].max()

    def neg_viterbi(self, T, L, prior=None):
        viterbi_output = self.viterbi(T, L)
        return (viterbi_output[0], -viterbi_output[1])

    def make_T(self, p_jump, p_grad):
        p_stay = 1 - (p_jump + p_grad)
        n_state = len(self.T_grad)
        T_stay = np.eye(n_state)
        T_jump = np.ones((n_state, n_state)) - T_stay
        T_jump /= T_jump.sum(1, keepdims=True)  # normalize
        T = p_stay * T_stay + p_grad * self.T_grad + p_jump * T_jump
        return T

    def optimize_jump_weight(self, L):
        def loss(x):
            T = self.make_T(x[0], x[1])
            log_likelihood = self.viterbi(np.log(T), L)[1]
            return -log_likelihood

        xs = [
            [i, 0]
            # for i in [0]
            for i in np.arange(0, 1.001, 0.02)
        ]

        losses = list(map(loss, xs))
        best = np.argmin(losses)
        return np.array(xs[best]), losses[best]

    def jump_prediction(self, L):
        """
        Returns the strategy sequence and its likelihood according to the jump model
        """
        weights, _ = self.optimize_jump_weight(L)
        T = self.make_T(*weights)
        predicted_sequence, neg_LL = self.neg_viterbi(np.log(T), L)
        return predicted_sequence, neg_LL, weights

    def apply_microscope(self, click_sequences, envs):
        """
        Returns the strategy sequence with the highest likelihood.

        Arguments:
            click_sequences: The nodes that are clicked in a given trial. Each click sequence should end with 0.
            Environments: The values under each node in the Mouselab-MDP graph.
            strategy_T: The temperature parameter for the strategies being inferred.
        Expects 2D lists for both click sequences and environments
        """
        num_sequences = len(click_sequences)
        if num_sequences == 0:
            return None, None, None
        L = np.array(self.compute_trials_likelihood(click_sequences, envs))
        weights = defaultdict(list)
        strategy_sequence, nll, weights = self.jump_prediction(L)
        strategies = [self.strategy_space[s] for s in strategy_sequence]
        return strategies, nll, weights

    def infer_sequences(
        self, click_sequences, envs, max_evals=100, fit_strategy_temperature=True
    ):
        if fit_strategy_temperature:
            s_param = hp.loguniform("strategy_T", np.log(1e-2), np.log(1e2))
            parameter_space = {"strategy_T": s_param}

            def nll(params):
                strategy_T = params["strategy_T"]
                self.set_strategy_T(strategy_T)
                _, objective_value, _ = self.apply_microscope(click_sequences, envs)
                return objective_value

            # algo = partial(tpe.suggest, n_startup_jobs=30)
            algo = tpe.suggest
            trials = generate_trials_to_calculate([{"strategy_T": 1}])
            # max evals should be greater than 1
            best_params = fmin(
                fn=nll,
                space=parameter_space,
                algo=algo,
                trials=trials,
                max_evals=max_evals - 1,
            )
            self.set_strategy_T(best_params["strategy_T"])
        strategies, nll, weights = self.apply_microscope(click_sequences, envs)
        return strategies, nll, weights, self.strategy_T

    def infer_participant_sequences(
        self,
        pids,
        p_envs,
        p_clicks,
        max_evals=100,
        fit_strategy_temperature=True,
        show_pids=True,
    ):
        strategies = defaultdict(list)
        temperatures = {}
        for pid in pids:
            if show_pids:
                print(pid)
            if pid in p_clicks:
                clicks = p_clicks[pid]
                envs = p_envs[pid]
                if len(clicks) == 0:
                    continue
                S, _, _, T = self.infer_sequences(
                    clicks,
                    envs,
                    fit_strategy_temperature=fit_strategy_temperature,
                    max_evals=max_evals,
                )
                strategies[pid] = S
                temperatures[pid] = T
                print("S AND T", S, T)
        return strategies, temperatures
