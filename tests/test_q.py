import json
import sys
from pathlib import Path

import numpy as np
import yaml
from costometer.utils import load_q_file
from mouselab.agents import Agent
from mouselab.cost_functions import *  # noqa: F401, F403
from mouselab.env_utils import get_ground_truths_from_json
from mouselab.graph_utils import annotate_mdp_graph, get_structure_properties
from mouselab.mouselab import MouselabEnv
from mouselab.policies import OptimalQ

sys.path.append(str(Path(__file__).parents[1].joinpath("cluster/src")))
from cluster_utils import create_test_env  # noqa : E402

if __name__ == "__main__":
    cost_function = "distance_graph_cost"
    cost_param_file = "test_params"

    cost_file = (
        Path(__file__)
        .resolve()
        .parents[1]
        .joinpath(f"data/inputs/yamls/cost_functions/{cost_function}.yaml")
    )

    with open(cost_file, "r") as f:
        cost_details = yaml.safe_load(f)

    if "cost_function" in cost_details:
        cost_function_name = cost_details["cost_function"]
        cost_function = eval(cost_details["cost_function"])
    else:
        cost_function_name = None
        cost_function = eval(cost_function)

    with open(
        Path(__file__)
        .resolve()
        .parents[1]
        .joinpath(f"cluster/parameters/cost/{cost_param_file}.txt"),
        "r",
    ) as f:
        all_cost_lines = f.read().splitlines()

    cost_parameters_settings = [
        {
            cost_parameter_arg: float(arg)
            for arg, cost_parameter_arg in zip(
                cost_line.split(","), cost_details["cost_parameter_args"]
            )
        }
        for cost_line in all_cost_lines
    ]

    for experiment_setting in [
        "reduced_leaf",
        "reduced_middle",
        "reduced_root",
        "reduced_variance",
    ]:
        for cost_parameter_setting in cost_parameters_settings:
            create_test_env(experiment_setting)

            q_dictionary = load_q_file(
                experiment_setting,
                cost_function=cost_function,
                cost_function_name=cost_function_name,
                cost_params=cost_parameter_setting,
                path=Path(__file__).parents[1].joinpath("cluster/data/q_files"),
            )

            policy = OptimalQ(preference=q_dictionary)

            # first we need to load experiment setting file and then structure file
            with open(
                Path(__file__)
                .parents[1]
                .joinpath(
                    f"data/inputs/yamls/experiment_settings/{experiment_setting}.yaml"
                ),
                "r",
            ) as stream:
                args = yaml.safe_load(stream)

            if "structure" in args:
                with open(
                    Path(__file__)
                    .parents[1]
                    .joinpath(
                        f"data/inputs/exp_inputs/structure/{args['structure']}.json"
                    ),
                    "rb",
                ) as f:
                    structure_data = json.load(f)

                structure_dicts = get_structure_properties(structure_data)
            else:
                structure_dicts = None

            if args["ground_truth_file"]:
                ground_truths = get_ground_truths_from_json(
                    Path(__file__)
                    .parents[1]
                    .joinpath(
                        f"data/inputs/exp_inputs/rewards/"
                        f"{args['ground_truth_file']}.json"
                    )
                )

            returns = []
            for ground_truth in ground_truths:
                agent = Agent()

                env = MouselabEnv.new_symmetric_registered(
                    experiment_setting=experiment_setting,
                    include_last_action=True,
                    cost=cost_function(**cost_parameter_setting),
                    ground_truth=ground_truth,
                )
                env.mdp_graph = annotate_mdp_graph(env.mdp_graph, structure_dicts)

                agent.register(env)

                agent.register(policy)
                policy_data = agent.run_many(10)

                returns.append(policy_data["return"])

            print(experiment_setting)
            print(np.mean(returns))
            print(
                [
                    q_dictionary[(init_state, action)]
                    for init_state in env.initial_states
                    for action in env.actions(init_state)
                ]
            )
