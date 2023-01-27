"""
Simulates trajectories, given some cost for RandomPolicy, SoftmaxPolicy \
or OptimalQ agents
"""
import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from cluster_utils import create_test_env, get_args_from_yamls
from costometer.agents.vanilla import SymmetricMouselabParticipant
from costometer.utils import get_param_string, load_q_file, traces_to_df
from mouselab.cost_functions import *  # noqa
from mouselab.envs.registry import registry
from mouselab.graph_utils import get_structure_properties
from mouselab.policies import OptimalQ, RandomPolicy, SoftmaxPolicy  # noqa
from scipy import stats  # noqa
import dill as pickle
from get_myopic_voc_values import get_state_action_values

if __name__ == "__main__":
    """
    Example calls:
    python src/simulate_trajectories.py -p SoftmaxPolicy -e high_increasing -c \
    linear_depth -v=1.0,10.0 -m test -n 3
    python src/simulate_trajectories.py -p RandomPolicy -e high_increasing
    python src/simulate_trajectories.py -p OptimalQ -e high_increasing -c linear_depth \
    -v=0.0,0.0
    """
    parser = ArgumentParser()
    parser.add_argument(
        "-p",
        "--policy",
        dest="policy",
        help="Policy to simulate",
        choices=["OptimalQ", "SoftmaxPolicy", "RandomPolicy"],
        default="SoftmaxPolicy",
        type=str,
    )
    parser.add_argument(
        "-e",
        "--experiment-setting",
        dest="experiment_setting",
        help="Experiment setting YAML file",
        default="high_increasing",
        type=str,
    )
    parser.add_argument(
        "-c",
        "--cost-function",
        dest="cost_function",
        help="Cost function YAML file",
        type=str,
        default="back_dist_depth_eff_forw",
    )
    parser.add_argument(
        "-b",
        "--bmps-file",
        dest="bmps_file",
        default="Myopic_VOC",
        help="BMPS Features and Optimization",
        type=str,
    )
    parser.add_argument(
        "-f",
        "--parameter-file",
        dest="parameter_file",
        default="participants",
        type=str,
    )
    parser.add_argument(
        "-t", "--num-trials", dest="num_trials", help="Num trials", type=int, default=20
    )
    parser.add_argument("-s", "--seed", dest="seed", help="seed", type=int, default=91)

    inputs = parser.parse_args()

    if inputs.cost_function:
        args = get_args_from_yamls(
            vars(inputs), attributes=["cost_function", "experiment_setting"]
        )
    else:
        args = get_args_from_yamls(vars(inputs), attributes=["experiment_setting"])
    path = Path(__file__).resolve().parents[2]

    experiment_setting = args["experiment_setting"]

    try:
        registry(experiment_setting)
    except:  # noqa: E722
        create_test_env(experiment_setting)

    if inputs.cost_function:
        cost_function = eval(args["cost_function"])
        if callable(cost_function):
            cost_function_name = inputs.cost_function
        else:
            cost_function_name = cost_function
    else:
        cost_function = None
        cost_function_name = None
        cost_parameters = {}

    with open(
        path.joinpath(
            f"data/inputs/exp_inputs/rewards/{args['ground_truth_file']}.json"
        ),
        "rb",
    ) as file_handler:
        ground_truths = json.load(file_handler)

    # if we have a ground truth file, set it
    ground_truth_file = args["ground_truth_file"]

    # make trajectory folders if they don't already exist
    path.joinpath(
        f"cluster/data/trajectories/{experiment_setting}/{inputs.policy}/"
    ).mkdir(parents=True, exist_ok=True)

    if "structure" in args:
        with open(
            path.joinpath(f"data/inputs/exp_inputs/structure/{args['structure']}.json"),
            "rb",
        ) as f:
            structure_data = json.load(f)

        structure_dicts = get_structure_properties(structure_data)
    else:
        structure_dicts = None

    with open(path.joinpath(f"cluster/parameters/simulations/{inputs.parameter_file}.pkl"), "rb") as f:
        possible_parameters = pickle.load(f)

    traces = []
    for fake_pid, curr_params in enumerate(possible_parameters):
        # separate gamma, kappa, cost, policy kwargs
        cost_parameters = {key: val for key, val in curr_params.items() if key in args["cost_parameter_args"]}
        policy_parameters = {key: val for key, val in curr_params.items() if key not in args["cost_parameter_args"] + ["gamma", "kappa"]}

        # set seed in policy_kwargs
        policy_parameters["seed"] = inputs.seed

        # construct q_function
        if inputs.cost_function:
                q_function = get_state_action_values(  # noqa : E731
                experiment_setting=args["experiment_setting"],
                bmps_file=inputs.bmps_file,
                cost_function=cost_function,
                cost_parameters=cost_parameters,
                structure=structure_dicts,
                env_params=args["env_params"],
                kappa=curr_params["kappa"],
                gamma=curr_params["gamma"],
                )
                policy_parameters["preference"] = q_function

        ground_truth_subsets = np.random.choice(
            ground_truths, inputs.num_trials, replace=False
        )

        if inputs.cost_function:
            additional_mouselab_kwargs = {
                "mdp_graph_properties": structure_dicts,
                **args["env_params"],
            }
        else:
            additional_mouselab_kwargs = {}

        simulated_participant = SymmetricMouselabParticipant(
            experiment_setting,
            policy_function=eval(inputs.policy),
            policy_kwargs=policy_parameters,
            num_trials=inputs.num_trials,
            cost_function=cost_function,
            cost_kwargs=cost_parameters,
            ground_truths=[trial["stateRewards"] for trial in ground_truth_subsets],
            trial_ids=[trial["trial_id"] for trial in ground_truth_subsets],
            additional_mouselab_kwargs=additional_mouselab_kwargs,
            kappa=curr_params["kappa"],
            gamma=curr_params["gamma"],
        )
        simulated_participant.simulate_trajectory()

        trace_df = traces_to_df([simulated_participant.trace])

        # add all information that might be useful
        for sim_param, sim_value in vars(inputs).items():
            trace_df[f"sim_{sim_param}"] = sim_value

        for cost_param, cost_val in curr_params.items():
            trace_df[f"sim_{cost_param}"] = cost_val

        trace_df["pid"] = fake_pid
        del trace_df["states"]

        traces.append(trace_df)

    full_df = pd.concat(traces)


    if inputs.cost_function is None:
        filename = path.joinpath(
            f"cluster/data/trajectories/{experiment_setting}/{inputs.policy}"
            f"/simulated_agents.csv"
        )
    else:
        filename = path.joinpath(
            f"cluster/data/trajectories/{experiment_setting}"
            f"/{inputs.policy}"
            f"/simulated_agents_{inputs.cost_function}_{inputs.parameter_file}.csv"
        )

    full_df.to_csv(filename, index=False)
