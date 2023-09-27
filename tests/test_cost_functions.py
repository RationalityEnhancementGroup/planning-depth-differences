import json
from pathlib import Path
from unittest import TestCase

import yaml
from mouselab.agents import Agent
from mouselab.cost_functions import *  # noqa : F401, F403
from mouselab.graph_utils import get_structure_properties
from mouselab.mouselab import MouselabEnv
from mouselab.policies import RandomPolicy
from toolz import curry


class TestCostFunctions(TestCase):
    EXPECTED_COSTS = {
        "back_added_cost": {1: -1, 2: -1, 3: 0, 11: 0, 10: 0, 9: 0},
        "depth_cost_weight": {1: -1, 2: -2, 3: -3, 11: -3, 10: -2, 9: -1},
        "distance_multiplier": {
            1: -1,
            2: -1,
            3: -1,
            11: -((1**2 + 3**2) ** (1 / 2)),
            10: -1,
            9: -1,
        },
        "forw_added_cost": {1: 0, 2: 0, 3: 0, 11: -1, 10: -1, 9: 0},
        "given_cost": {1: -1, 2: -1, 3: -1, 11: -1, 10: -1, 9: -1},
    }

    @classmethod
    def setUpClass(cls) -> None:
        cls.data_path = Path(__file__).resolve().parents[1] / "data"

        cost_file = (
            cls.data_path
            / "inputs"
            / "yamls"
            / "cost_functions"
            / "back_dist_depth_eff_forw.yaml"
        )

        with open(cost_file, "r") as f:
            cls.cost_details = yaml.safe_load(f)

        experiment_setting_files = (
            cls.data_path / "inputs" / "yamls" / "experiment_settings"
        ).glob("*.yaml")

        cls.experiment_settings = {}
        for experiment_setting_file in experiment_setting_files:
            with open(experiment_setting_file, "r") as f:
                cls.experiment_settings[experiment_setting_file.stem] = yaml.safe_load(
                    f
                )

                if "structure" in cls.experiment_settings[experiment_setting_file.stem]:
                    structure_file = (
                        cls.data_path
                        / "inputs"
                        / "exp_inputs"
                        / "structure"
                        / f"{cls.experiment_settings[experiment_setting_file.stem]['structure']}.json"  # noqa : E501
                    )
                    with open(
                        structure_file,
                        "rb",
                    ) as f:
                        structure_data = json.load(f)

                    cls.experiment_settings[experiment_setting_file.stem][
                        "structure_dicts"
                    ] = get_structure_properties(structure_data)
                else:
                    cls.experiment_settings[experiment_setting_file.stem][
                        "structure_dicts"
                    ] = None

    @staticmethod
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
            experiment_setting,
            cost=curr_cost_function,
            **curr_cost_details["env_params"],
            mdp_graph_properties=experiment_settings[experiment_setting][
                "structure_dicts"
            ],
        )

        return curr_env

    def test_cost(self):
        for cost_parameter in self.cost_details["cost_parameter_args"]:
            env = self.create_env_with_cost(
                {
                    p: int(p == cost_parameter)
                    for p in self.cost_details["cost_parameter_args"]
                },
                static_cost_dictionary={},
                cost_function=eval(self.cost_details["cost_function"]),
                experiment_settings=self.experiment_settings,
                experiment_setting="high_increasing",
                curr_cost_details=self.cost_details,
            )

            for action, result in self.EXPECTED_COSTS[cost_parameter].items():
                self.assertEqual(env.step(action)[1], result)

    def test_trajectory_rewards(self):
        for experiment_setting in self.experiment_settings:
            cost_function = eval(self.cost_details["cost_function"])

            curr_env = self.create_env_with_cost(
                static_cost_dictionary={},
                cost_function=cost_function,
                experiment_settings=self.experiment_settings,
                experiment_setting=experiment_setting,
                curr_cost_details=self.cost_details,
            )

            for cost_parameter_val in [2, 5, 10]:
                mouselab_env = curr_env(
                    cost_dictionary={
                        param: cost_parameter_val
                        for param in self.cost_details["cost_parameter_args"]
                    }
                )

                agent = Agent()
                agent.register(mouselab_env)

                agent.register(RandomPolicy())

                trace = agent.run_many(num_episodes=10)

                mouselab_env.ground_truth = mouselab_env.ground_truth
                rewards = []
                for episode in trace["actions"]:
                    mouselab_env.reset()
                    curr_episode_rewards = []
                    for action in episode:
                        _, reward, _, _ = mouselab_env.step(action)
                        curr_episode_rewards.append(reward)
                    rewards.append(curr_episode_rewards)

                self.assertEqual(trace["rewards"], rewards)
