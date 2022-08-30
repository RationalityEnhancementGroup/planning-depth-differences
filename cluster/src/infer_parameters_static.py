"""
This scripts calculates the probability of a trace coming from SoftmaxPolicy agents \
(who are not learning!)  given some cost function, values of cost weights and \
temperatures or a RandomPolicy agent.
"""
import json
from argparse import ArgumentParser
from pathlib import Path

import yaml
from cluster_utils import (
    get_args_from_yamls,
    get_human_trajectories,
    get_simulated_trajectories,
)
from costometer.agents.vanilla import SymmetricMouselabParticipant
from costometer.inference import GridInference
from costometer.utils import get_param_string, get_temp_prior
from mouselab.cost_functions import *  # noqa
from mouselab.distributions import Categorical
from mouselab.graph_utils import get_structure_properties
from mouselab.policies import RandomPolicy, SoftmaxPolicy
from scipy import stats  # noqa

if __name__ == "__main__":
    """# noqa
    Examples:
    python infer_parameters_static_ray.py -e CogSciPoster -c cost_function -t temperature_file -v params_full
    python infer_parameters_static_ray.py -e data/trajectories/{experiment_setting}/{policy_function}/simulated_agents_{cost_function}*  -c cost_function -t temperature_file -v params_full
    """
    # get arguments
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--experiment",
        dest="experiment",
        help="Experiment",
        type=str,
    )
    parser.add_argument(
        "-c",
        "--cost-function",
        dest="cost_function",
        help="Cost function YAML file",
        type=str,
    )
    parser.add_argument(
        "-t",
        "--temperature-file",
        dest="temperature_file",
        help="File with temperatures to infer over",
        type=str,
    )
    parser.add_argument(
        "-v",
        "--values",
        dest="cost_parameter_values",
        help="Cost parameter values as comma separated string, e.g. '1.00,2.00'",
        type=str,
    )

    inputs = parser.parse_args()

    if "*" in inputs.experiment or ".csv" in inputs.experiment:
        args = get_args_from_yamls(vars(inputs), attributes=["cost_function"])
        args["experiment"] = inputs.experiment
        args["experiment_setting"] = inputs.experiment.split("/")[-3]
    else:
        args = get_args_from_yamls(
            vars(inputs), attributes=["cost_function", "experiment"]
        )

    path = Path(__file__).resolve().parents[2]

    # if wild card or .csv in experiment name, this is file pattern for
    # simulated trajectories
    if "*" in args["experiment"] or ".csv" in args["experiment"]:
        traces = get_simulated_trajectories(
            args["experiment"],
            args["experiment_setting"],
            simulated_trajectory_path=path.joinpath("cluster"),
        )
        experiment_folder = "simulated/" + "/".join(
            args["experiment"].split("/")[-3:-1]
        )
        # simulation params = file name, without asterisk or extension
        simulation_params = "_" + args["experiment"].split("/")[-1].replace(
            "*", ""
        ).replace(".csv", "")
    else:
        traces = get_human_trajectories(
            args["experiment"],
            include_last_action=args["env_params"]["include_last_action"],
        )
        experiment_folder = args["experiment"]
        # data not simulated, no simulation params
        simulation_params = ""

    cost_parameters = {
        cost_parameter_arg: Categorical([float(arg)], [1])
        for arg, cost_parameter_arg in zip(
            inputs.cost_parameter_values.split(","), args["cost_parameter_args"]
        )
    }
    cost_parameter_dict = {
        cost_parameter_arg: float(arg)
        for arg, cost_parameter_arg in zip(
            inputs.cost_parameter_values.split(","), args["cost_parameter_args"]
        )
    }

    if inputs.temperature_file is not None:
        yaml_path = str(
            path.joinpath(
                f"data/inputs/yamls/temperatures/{inputs.temperature_file}.yaml"
            )
        )
        with open(str(yaml_path), "r") as stream:
            prior_inputs = yaml.safe_load(stream)
        priors = get_temp_prior(
            rv=eval(prior_inputs["rv"]),
            possible_vals=prior_inputs["possible_temps"],
            inverse=prior_inputs["inverse"],
        )
    else:
        priors = None

    # make experiment folder if it doesn't already exist
    path.joinpath(
        f"cluster/data/logliks/{args['cost_function']}/{experiment_folder}"
    ).mkdir(parents=True, exist_ok=True)

    if "structure" in args:
        with open(
            Path(__file__)
            .parents[2]
            .joinpath(f"data/inputs/exp_inputs/structure/{args['structure']}.json"),
            "rb",
        ) as f:
            structure_data = json.load(f)

        structure_dicts = get_structure_properties(structure_data)
    else:
        structure_dicts = None

    cost_function = eval(args["cost_function"])
    if callable(eval(args["cost_function"])):
        cost_function_name = inputs.cost_function
    else:
        cost_function_name = None

    softmax_ray_object = GridInference(
        traces=traces,
        participant_class=SymmetricMouselabParticipant,
        participant_kwargs={
            "experiment_setting": args["experiment_setting"],
            "policy_function": SoftmaxPolicy,
            # "additional_mouselab_kwargs": {"mdp_graph_properties": structure_dicts},
        },
        held_constant_policy_kwargs={
            "noise": 0,
            "q_path": path.joinpath("cluster/data/q_files"),
        },
        policy_parameters={"temp": priors},
        cost_function=cost_function,
        cost_parameters=cost_parameters,
        cost_function_name=cost_function_name,
    )

    softmax_ray_object.run()

    optimization_results = softmax_ray_object.get_optimization_results()
    filename = path.joinpath(
        f"cluster/data/logliks/{cost_function.__name__}/{experiment_folder}/"
        f"SoftmaxPolicy_optimization_results_{get_param_string(cost_parameter_dict)}"
        f"{simulation_params}.csv"
    )
    optimization_results.to_csv(filename, index=False)

    random_ray_object = GridInference(
        traces=traces,
        participant_class=SymmetricMouselabParticipant,
        participant_kwargs={
            "experiment_setting": args["experiment_setting"],
            "policy_function": RandomPolicy,
            # "additional_mouselab_kwargs": {"mdp_graph_properties": structure_dicts},
        },
        cost_function=cost_function,
        cost_parameters=cost_parameters,
        cost_function_name=cost_function_name,
    )

    random_ray_object.run()

    optimization_results = random_ray_object.get_optimization_results()
    filename = path.joinpath(
        f"cluster/data/logliks/{cost_function_name}/{experiment_folder}/"
        f"RandomPolicy_optimization_results{simulation_params}.csv"
    )
    optimization_results.to_csv(filename, index=False)
