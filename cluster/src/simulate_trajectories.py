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
from mouselab.graph_utils import get_structure_properties
from mouselab.policies import OptimalQ, RandomPolicy, SoftmaxPolicy  # noqa
from scipy import stats  # noqa

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
        type=str,
    )
    parser.add_argument(
        "-e",
        "--experiment-setting",
        dest="experiment_setting",
        help="Experiment setting YAML file",
        type=str,
    )
    parser.add_argument(
        "-x",
        "--exact",
        dest="exact",
        help="Use exact Q values instead of approximate values",
        default=True,
        action="store_false",
    )
    parser.add_argument(
        "-c",
        "--cost-function",
        dest="cost_function",
        help="Cost function YAML file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-m",
        "--temperature-file",
        dest="temperature_file",
        help="File with temperatures to infer over",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-v",
        "--values",
        dest="cost_parameter_values",
        help="Cost parameter values as comma separated string, e.g. '1.00,2.00'",
        type=str,
    )
    parser.add_argument(
        "-n",
        "--num-simulated",
        dest="num_simulated",
        help="Num simulations",
        type=int,
        default=200,
    )
    parser.add_argument(
        "-t", "--num-trials", dest="num_trials", help="Num trials", type=int, default=30
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

    policy_kwargs = {"seed": inputs.seed}

    experiment_setting = args["experiment_setting"]

    # test setting unique to this work
    if experiment_setting in [
        "small_test_case",
        "reduced_leaf",
        "reduced_middle",
        "reduced_root",
        "reduced_variance",
    ]:
        create_test_env(experiment_setting)

    if inputs.cost_function:
        cost_function = eval(args["cost_function"])
        try:
            cost_parameters = {
                cost_parameter_arg: float(arg)
                for arg, cost_parameter_arg in zip(
                    inputs.cost_parameter_values.split(","), args["cost_parameter_args"]
                )
            }
        except ValueError as e:
            raise e
        q_dictionary = load_q_file(
            experiment_setting,
            cost_function=cost_function,
            cost_params=cost_parameters,
            path=path.joinpath("cluster/data/bmps/preferences")
            if not inputs.exact
            else path.joinpath("cluster/data/q_files"),
        )
        policy_kwargs["preference"] = q_dictionary
    else:
        cost_function = None
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

    if inputs.temperature_file is not None:
        temperatures = np.loadtxt(
            path.joinpath(
                f"cluster/parameters/temperatures/{inputs.temperature_file}.txt"
            )
        )
        possible_parameters = [
            {**policy_kwargs, "temp": temp, "noise": 0} for temp in temperatures
        ]
    else:
        possible_parameters = [policy_kwargs]

    traces = []
    fake_pid = 0
    for simulation in range(inputs.num_simulated):
        ground_truth_subsets = np.random.choice(
            ground_truths, inputs.num_trials, replace=False
        )
        for possible_parameter in possible_parameters:
            simulated_participant = SymmetricMouselabParticipant(
                experiment_setting,
                policy_function=eval(inputs.policy),
                policy_kwargs=possible_parameter,
                num_trials=inputs.num_trials,
                cost_function=cost_function,
                cost_kwargs=cost_parameters,
                ground_truths=[trial["stateRewards"] for trial in ground_truth_subsets],
                trial_ids=[trial["trial_id"] for trial in ground_truth_subsets],
                additional_mouselab_kwargs={
                    "mdp_graph_properties": structure_dicts,
                    **args["env_params"],
                },
            )
            simulated_participant.simulate_trajectory()

            trace_df = traces_to_df([simulated_participant.trace])

            # add all information that might be useful
            for sim_param, sim_value in vars(inputs).items():
                trace_df[f"sim_{sim_param}"] = sim_value

            for cost_param, cost_val in cost_parameters.items():
                trace_df[f"sim_{cost_param}"] = cost_val

            for policy_param, policy_val in possible_parameter.items():
                if policy_param != "preference":
                    trace_df[f"sim_{policy_param}"] = policy_val

            trace_df["pid"] = fake_pid

            fake_pid += 1

            traces.append(trace_df)

    full_df = pd.concat(traces)

    parameter_string = get_param_string(cost_parameters)
    policy_params = []
    for key, param in sorted(policy_kwargs.items()):
        if isinstance(param, dict):
            pass
        elif isinstance(param, str):
            policy_params.append(param)
        else:
            policy_params.append(f"{param:.2f}")
    policy_string = "_".join(policy_params)

    if inputs.cost_function is None:
        filename = path.joinpath(
            f"cluster/data/trajectories/{experiment_setting}/{inputs.policy}"
            f"/simulated_agents_{policy_string}.csv"
        )
    else:
        filename = path.joinpath(
            f"cluster/data/trajectories/{experiment_setting}/{inputs.policy}"
            f"/simulated_agents_{inputs.cost_function}_{parameter_string}"
            f"_{policy_string}.csv"
        )

    full_df.to_csv(filename)
