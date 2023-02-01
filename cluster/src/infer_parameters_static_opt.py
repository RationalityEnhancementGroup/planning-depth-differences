"""
This scripts calculates the probability of a trace coming from SoftmaxPolicy agents \
(who are not learning!)  given some cost function, values of cost weights and \
temperatures or a RandomPolicy agent.
It uses hyperopt.
"""
import json
import time
from argparse import ArgumentParser
from pathlib import Path

import dill as pickle
import numpy as np
import yaml
from cluster_utils import (
    get_args_from_yamls,
    get_human_trajectories,
    get_simulated_trajectories,
)
from costometer.agents.vanilla import SymmetricMouselabParticipant
from costometer.inference import HyperoptOptimizerInference
from costometer.utils import get_temp_prior
from get_myopic_voc_values import get_state_action_values
from mouselab.cost_functions import *  # noqa
from mouselab.graph_utils import get_structure_properties
from mouselab.policies import SoftmaxPolicy
from scipy import stats  # noqa

if __name__ == "__main__":
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
        default="expon",
        dest="temperature_file",
        help="File with temperatures to infer over",
        type=str,
    )
    parser.add_argument(
        "-p",
        "--prior-file",
        dest="prior_file",
        help="File with priors and search space",
        type=str,
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
        "-n",
        "--num-evals",
        dest="num_evals",
        type=int,
    )
    parser.add_argument(
        "-d",
        "--pid",
        dest="pid",
        type=int,
    )
    parser.add_argument(
        "-k",
        "--block",
        dest="block",
        default=None,
        help="Block",
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
        args = {**args, **get_args_from_yamls(args, attributes=["experiment_setting"])}

    path = Path(__file__).resolve().parents[2]

    # if wild card or .csv in experiment name, this is file pattern for
    # simulated trajectories
    if "*" in args["experiment"] or ".csv" in args["experiment"]:
        traces = get_simulated_trajectories(
            args["experiment"],
            args["experiment_setting"],
            pids=[inputs.pid],
            simulated_trajectory_path=path.joinpath("cluster"),
        )
        experiment_folder = "simulated/" + "/".join(
            args["experiment"].split("/")[-3:-1]
        )
        simulation_params = "_" + args["experiment"].split("/")[-1].replace(
            "*", ""
        ).replace(".csv", "")
    else:
        if inputs.block:
            block = [inputs.block]
        else:
            block = inputs.block

        traces = get_human_trajectories(
            args["experiment"],
            pids=[inputs.pid],
            blocks=block,
            include_last_action=args["env_params"]["include_last_action"],
        )
        experiment_folder = args["experiment"]

        if inputs.block != "test":
            simulation_params = "_" + inputs.block
        else:
            simulation_params = ""

    if inputs.prior_file is not None:
        yaml_path = str(
            path.joinpath(f"cluster/parameters/priors/{inputs.prior_file}.yaml")
        )
        with open(str(yaml_path), "r") as stream:
            prior_inputs = yaml.safe_load(stream)
    else:
        prior_inputs = None

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

    if inputs.temperature_file is not None:
        yaml_path = str(
            path.joinpath(
                f"data/inputs/yamls/temperatures/{inputs.temperature_file}.yaml"
            )
        )
        with open(str(yaml_path), "r") as stream:
            temp_priors = yaml.safe_load(stream)
        temp_priors = get_temp_prior(
            rv=eval(temp_priors["rv"]),
            possible_vals=temp_priors["possible_temps"],
            inverse=temp_priors["inverse"],
        )
        temp_prior_dict = dict(zip(temp_priors.vals, temp_priors.probs))
        prior_inputs["policy_parameters"]["temp"]["prior"] = lambda val: np.log(
            temp_prior_dict[val]
        )
    else:
        temp_priors = None

    cost_function = eval(args["cost_function"])
    if callable(eval(args["cost_function"])):
        cost_function_name = inputs.cost_function
    else:
        cost_function_name = None

    q_function_generator = (
        lambda cost_parameters, a, g: get_state_action_values(  # noqa
            experiment_setting=args["experiment_setting"],
            bmps_file=inputs.bmps_file,
            cost_function=cost_function,
            cost_parameters=cost_parameters,
            structure=structure_dicts,
            env_params=args["env_params"],
            bmps_path=Path(__file__).parents[1].joinpath("parameters/bmps/"),
            kappa=a,
            gamma=g,
        )
    )

    softmax_opt_object = HyperoptOptimizerInference(
        traces=traces,
        participant_class=SymmetricMouselabParticipant,
        participant_kwargs={
            "experiment_setting": args["experiment_setting"],
            "policy_function": SoftmaxPolicy,
            "additional_mouselab_kwargs": {"mdp_graph_properties": structure_dicts},
        },
        held_constant_policy_kwargs={
            "noise": 0,
            "q_function_generator": q_function_generator,
        },
        held_constant_cost_kwargs={},
        policy_parameters=prior_inputs["policy_parameters"],
        cost_function=cost_function,
        cost_parameters=prior_inputs["cost_parameters"],
        optimization_settings={"verbose": 0, "max_evals": inputs.num_evals},
    )

    softmax_opt_object.run()
    all_results = softmax_opt_object.get_optimization_results()

    # make experiment folder if it doesn't already exist
    path.joinpath(
        f"cluster/data/logliks_opt/{cost_function_name}/{experiment_folder}"
    ).mkdir(parents=True, exist_ok=True)

    filename = path.joinpath(
        f"cluster/data/logliks_opt/{cost_function_name}/{experiment_folder}/"
        f"SoftmaxPolicy_optimization_results"
        f"{simulation_params}_{inputs.pid}_"
        f"{inputs.num_evals}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    )
    with open(filename, "wb") as f:
        pickle.dump(all_results, f)
