"""
This scripts calculates the probability of a trace coming from SoftmaxPolicy agents \
(who are not learning!)  given some cost function, values of cost weights and \
temperatures or a RandomPolicy agent.
"""
import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np  # noqa, for the pickled Q value function
import yaml
from cluster_utils import (
    create_test_env,
    get_args_from_yamls,
    get_human_trajectories,
    get_simulated_trajectories,
)
from costometer.agents.vanilla import SymmetricMouselabParticipant
from costometer.inference import GridInference
from costometer.utils import get_param_string, get_temp_prior
from get_myopic_voc_values import get_state_action_values
from mouselab.cost_functions import *  # noqa
from mouselab.distributions import Categorical
from mouselab.envs.registry import registry
from mouselab.graph_utils import get_structure_properties
from mouselab.policies import RandomPolicy, SoftmaxPolicy
from scipy import stats  # noqa, for the temp prior

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
        "-x",
        "--exact",
        dest="exact",
        help="Use exact Q values instead of approximate values",
        default=False,
        action="store_true",
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
        "-p",
        "--participant-subset-file",
        dest="participant_subset_file",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-k",
        "--block",
        dest="block",
        default=None,
        help="Block",
        type=str,
    )
    parser.add_argument(
        "-v",
        "--values",
        dest="cost_parameter_values",
        help="Cost parameter values as comma separated string, e.g. '1.00,2.00'",
        type=str,
    )
    parser.add_argument(
        "-g",
        "--gamma_file",
        dest="gamma_file",
        help="gamma_file",
        type=str,
        default="full",
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
        "-a",
        "--kappa_file",
        dest="kappa_file",
        help="kappa_file",
        type=str,
        default="full",
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

    args = {
        **args,
        **get_args_from_yamls(
            {"experiment_setting": args["experiment_setting"]},
            attributes=["experiment_setting"],
        ),
    }

    try:
        registry(args["experiment_setting"])
    except:  # noqa: E722
        create_test_env(args["experiment_setting"])

    if "structure" in args:
        with open(
            path.joinpath(f"data/inputs/exp_inputs/structure/{args['structure']}.json"),
            "rb",
        ) as f:
            structure_data = json.load(f)

        structure_dicts = get_structure_properties(structure_data)
    else:
        structure_dicts = None

    # if wild card or .csv in experiment name, this is file pattern for
    # simulated trajectories
    if "*" in args["experiment"] or ".csv" in args["experiment"]:
        traces = get_simulated_trajectories(
            args["experiment"],
            args["experiment_setting"],
            simulated_trajectory_path=path.joinpath("cluster"),
            additional_mouselab_kwargs={
                "mdp_graph_properties": structure_dicts,
                **args["env_params"],
            },
        )
        experiment_folder = "simulated/" + "/".join(
            args["experiment"].split("/")[-3:-1]
        )
        # simulation params = file name, without asterisk or extension
        simulation_params = "_" + args["experiment"].split("/")[-1].replace(
            "*", ""
        ).replace(".csv", "")
    else:
        if inputs.participant_subset_file:
            pids = (
                Path(__file__)
                .resolve()
                .parents[1]
                .joinpath(f"parameters/pids/" f"{inputs.participant_subset_file}.txt")
            )

            with open(pids, "r") as f:
                pids = [int(pid) for pid in f.read().splitlines()]

        else:
            pids = None

        if inputs.block:
            block = [inputs.block]
        else:
            block = inputs.block

        traces = get_human_trajectories(
            inputs.experiment,
            pids=pids,
            blocks=block,
            include_last_action=args["env_params"]["include_last_action"],
        )
        experiment_folder = args["experiment"]
        # data not simulated, no simulation params

        if inputs.participant_subset_file:
            simulation_params = "_" + inputs.participant_subset_file
        else:
            simulation_params = ""

        if inputs.block != "test":
            simulation_params = simulation_params + "_" + inputs.block

    cost_parameter_dict = {
        cost_parameter_arg: arg
        for arg, cost_parameter_arg in zip(
            inputs.cost_parameter_values.split(","), args["cost_parameter_args"]
        )
    }

    cost_parameters = {}
    for arg, cost_parameter_arg in zip(
        inputs.cost_parameter_values.split(","), args["cost_parameter_args"]
    ):
        cost_parameters[cost_parameter_arg] = Categorical([float(arg)], [1])

    if inputs.temperature_file is not None:
        yaml_path = str(
            path.joinpath(
                f"data/inputs/yamls/temperatures/{inputs.temperature_file}.yaml"
            )
        )
        with open(str(yaml_path), "r") as stream:
            prior_inputs = yaml.safe_load(stream)
        temp_priors = get_temp_prior(
            rv=eval(prior_inputs["rv"]),
            possible_vals=prior_inputs["possible_temps"],
            inverse=prior_inputs["inverse"],
        )
    else:
        temp_priors = None

    cost_function = eval(args["cost_function"])
    if callable(eval(args["cost_function"])):
        cost_function_name = inputs.cost_function
    else:
        cost_function_name = None

    with open(
        path.joinpath(f"cluster/parameters/gammas/{inputs.gamma_file}.txt"), "r"
    ) as f:
        gamma_values = [float(val) for val in f.read().splitlines()]

    gamma_priors = Categorical(
        gamma_values, [1 / len(gamma_values)] * len(gamma_values)
    )

    with open(
        path.joinpath(f"cluster/parameters/kappas/{inputs.kappa_file}.txt"), "r"
    ) as f:
        kappa_values = [float(val) for val in f.read().splitlines()]

    kappa_priors = Categorical(
        kappa_values, [1 / len(kappa_values)] * len(kappa_values)
    )

    q_function_generator = (
        lambda cost_parameters, a, g: get_state_action_values(  # noqa : E731
            experiment_setting=args["experiment_setting"],
            bmps_file=inputs.bmps_file,
            cost_function=cost_function,
            cost_parameters=cost_parameters,
            structure=structure_dicts,
            env_params=args["env_params"],
            kappa=a,
            gamma=g,
        )
    )

    # make experiment folder if it doesn't already exist
    path.joinpath(
        f"cluster/data/logliks/{cost_function_name}/{experiment_folder}/"
    ).mkdir(parents=True, exist_ok=True)

    softmax_filename = path.joinpath(
        f"cluster/data/logliks/{cost_function_name}/{experiment_folder}/"
        f"SoftmaxPolicy_optimization_results_{get_param_string(cost_parameter_dict)}"
        f"{simulation_params}.csv"
    )

    if not softmax_filename.exists():
        softmax_ray_object = GridInference(
            traces=traces,
            participant_class=SymmetricMouselabParticipant,
            participant_kwargs={
                "experiment_setting": args["experiment_setting"],
                "policy_function": SoftmaxPolicy,
                "additional_mouselab_kwargs": {
                    "mdp_graph_properties": structure_dicts,
                    **args["env_params"],
                },
            },
            held_constant_policy_kwargs={
                "noise": 0,
                "q_function_generator": q_function_generator,
            },
            policy_parameters={
                "temp": temp_priors,
                "kappa": kappa_priors,
                "gamma": gamma_priors,
            },
            cost_function=cost_function,
            cost_parameters=cost_parameters,
            cost_function_name=cost_function_name,
        )

        softmax_ray_object.run()

        optimization_results = softmax_ray_object.get_optimization_results()

        optimization_results.to_csv(softmax_filename, index=False)

    random_filename = path.joinpath(
        f"cluster/data/logliks/{cost_function_name}/{experiment_folder}/"
        f"RandomPolicy_optimization_results{simulation_params}.csv"
    )
    if not random_filename.exists():
        random_ray_object = GridInference(
            traces=traces,
            participant_class=SymmetricMouselabParticipant,
            participant_kwargs={
                "experiment_setting": args["experiment_setting"],
                "policy_function": RandomPolicy,
                "additional_mouselab_kwargs": {
                    "mdp_graph_properties": structure_dicts,
                    **args["env_params"],
                },
            },
            held_constant_policy_kwargs={
                "kappa": 1,
                "gamma": 1,
            },
            cost_function=cost_function,
            cost_parameters={
                cost_parameter_arg: Categorical(
                    [args["constant_values"][cost_parameter_arg]], [1]
                )
                for cost_parameter_arg in args["cost_parameter_args"]
            },
            cost_function_name=cost_function_name,
        )

        random_ray_object.run()

        optimization_results = random_ray_object.get_optimization_results()
        optimization_results.to_csv(random_filename, index=False)
