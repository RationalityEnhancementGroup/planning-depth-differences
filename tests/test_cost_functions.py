import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml
from mouselab.agents import Agent
from mouselab.cost_functions import *  # noqa : F401, F403
from mouselab.graph_utils import annotate_mdp_graph, get_structure_properties
from mouselab.mouselab import MouselabEnv
from mouselab.policies import RandomPolicy
from toolz import curry


@curry
def create_env_with_cost(
    cost_dictionary,
    static_cost_dictionary,
    cost_function,
    experiment_settings,
    experiment_setting,
    curr_cost_details,
):
    curr_cost_function = cost_function(**cost_dictionary, **static_cost_dictionary)
    curr_env = MouselabEnv.new_symmetric_registered(
        experiment_setting, cost=curr_cost_function, **curr_cost_details["env_params"]
    )

    curr_env.mdp_graph = annotate_mdp_graph(
        curr_env.mdp_graph, experiment_settings[experiment_setting]["structure_dicts"]
    )
    return curr_env


if __name__ == "__main__":
    cost_files = (
        Path(__file__)
        .resolve()
        .parents[1]
        .glob("data/inputs/yamls/cost_functions/*.yaml")
    )

    cost_details = {}
    for cost_file in cost_files:
        with open(cost_file, "r") as f:
            cost_details[cost_file.stem] = yaml.safe_load(f)

    experiment_setting_files = (
        Path(__file__)
        .resolve()
        .parents[1]
        .glob("data/inputs/yamls/experiment_settings/*.yaml")
    )

    experiment_settings = {}
    for experiment_setting_file in experiment_setting_files:
        with open(experiment_setting_file, "r") as f:
            experiment_settings[experiment_setting_file.stem] = yaml.safe_load(f)

            if "structure" in experiment_settings[experiment_setting_file.stem]:
                with open(
                    Path(__file__)
                    .parents[1]
                    .joinpath(
                        f"data/inputs/exp_inputs/structure/"
                        f"{experiment_settings[experiment_setting_file.stem]['structure']}.json"  # noqa : E501
                    ),
                    "rb",
                ) as f:
                    structure_data = json.load(f)

                experiment_settings[experiment_setting_file.stem][
                    "structure_dicts"
                ] = get_structure_properties(structure_data)
            else:
                experiment_settings[experiment_setting_file.stem][
                    "structure_dicts"
                ] = None

    experiment_setting = "high_increasing"

    cost_envs = {}

    for cost_name, curr_cost_details in cost_details.items():
        if "cost_function" in curr_cost_details:
            cost_function = eval(curr_cost_details["cost_function"])
        else:
            cost_function = eval(cost_name)

        for model, model_name in eval(curr_cost_details["model_name"]).items():
            if len(model) < 2:
                constant_parameter = {
                    cost_parameter_arg: curr_cost_details["constant_values"][
                        cost_parameter_arg
                    ]
                    for cost_parameter_arg in model
                }
                varied_parameters = [
                    cost_parameter_arg
                    for cost_parameter_arg in curr_cost_details["cost_parameter_args"]
                    if cost_parameter_arg not in constant_parameter.keys()
                ]
                curr_env = create_env_with_cost(
                    static_cost_dictionary=constant_parameter,
                    cost_function=cost_function,
                    experiment_settings=experiment_settings,
                    experiment_setting=experiment_setting,
                    curr_cost_details=curr_cost_details,
                )

                if model_name not in cost_envs:
                    cost_envs[model_name] = [
                        (
                            varied_parameters,
                            deepcopy(curr_env),
                            cost_name,
                            cost_function,
                        )
                    ]
                else:
                    cost_envs[model_name].append(
                        (
                            varied_parameters,
                            deepcopy(curr_env),
                            cost_name,
                            cost_function,
                        )
                    )

        for model_name, envs in cost_envs.items():
            print(model_name)
            if len(envs) > 1:
                for cost_parameter_val in [2, 5, 10]:
                    mouselab_envs = [
                        env[1]({param: cost_parameter_val for param in env[0]})
                        for env in envs
                    ]

                agent = Agent()
                agent.register(mouselab_envs[0])

                agent.register(RandomPolicy())

                trace = agent.run_many(num_episodes=10)

                for mouselab_env in mouselab_envs[1:]:
                    mouselab_env.ground_truth = mouselab_envs[0].ground_truth
                    rewards = []
                    for episode in trace["actions"]:
                        mouselab_env.reset()
                        curr_episode_rewards = []
                        for action in episode:
                            _, reward, _, _ = mouselab_env.step(action)
                            curr_episode_rewards.append(reward)
                        rewards.append(curr_episode_rewards)

                    assert np.all(trace["rewards"] == np.asarray(rewards))
