import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from mouselab.env_utils import (
    get_all_possible_ground_truths,
    get_all_possible_sa_pairs_for_env,
)
from mouselab.exact import hash_tree
from mouselab.graph_utils import annotate_mdp_graph, get_structure_properties
from mouselab.mouselab import MouselabEnv

sys.path.append(str(Path(__file__).parents[1].joinpath("cluster/src")))
from cluster_utils import create_test_env

if __name__ == "__main__":
    experiment_setting = "high_increasing"
    include_last_action = True

    # first we need to load experiment setting file and then structure file
    with open(
        Path(__file__)
        .parents[1]
        .joinpath(f"data/inputs/yamls/experiment_settings/{experiment_setting}.yaml"),
        "r",
    ) as stream:
        args = yaml.safe_load(stream)

    if "structure" in args:
        with open(
            Path(__file__)
            .parents[1]
            .joinpath(f"data/inputs/exp_inputs/structure/{args['structure']}.json"),
            "rb",
        ) as f:
            structure_data = json.load(f)

        structure_dicts = get_structure_properties(structure_data)
    else:
        structure_dicts = None

    ground_truth = [0, -4, -4, -24, 24, -4, 4, 24, 24, -4, -4, -24, 48]

    env = MouselabEnv.new_symmetric_registered(
        experiment_setting=experiment_setting,
        include_last_action=include_last_action,
        ground_truth=ground_truth,
    )

    env.mdp_graph = annotate_mdp_graph(env.mdp_graph, structure_dicts)

    for action in [1, 5, 9]:
        env.reset()
        env.step(action)
        print(env._state)
        plt.figure()
        plt.title(str(hash_tree(env, env._state))[-6:])
        env._render(use_networkx=True)
        for action in env.actions(env._state):
            if action != env.term_action:
                plt.text(
                    *env.mdp_graph.nodes[action]["layout"],
                    str(hash_tree(env, env._state, action=action))[-6:],
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=8,
                )
        plt.show()

    # for experiment_setting in [
    #     "small_test_case",
    #     # "reduced_leaf",
    #     # "high_increasing",
    #     "reduced_middle",
    #     "reduced_root",
    #     "reduced_variance",
    # ]:
    #     create_test_env(experiment_setting)
    #     include_last_action = True
    #
    #     env = MouselabEnv.new_symmetric_registered(
    #         experiment_setting=experiment_setting, include_last_action=include_last_action
    #     )
    #
    #     all_possible_states = set(state for state, action in get_all_possible_sa_pairs_for_env(env))
    #     # all_possible_states = list(get_all_possible_ground_truths(env))
    #     hash_states = set([(hash_tree(env, state), state) for state in all_possible_states])
    #
    #     print(experiment_setting, include_last_action)
    #     print(len(hash_states), len(all_possible_states))
    #     print(len(all_possible_states))
