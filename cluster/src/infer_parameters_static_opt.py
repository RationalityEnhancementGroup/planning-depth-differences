"""
This scripts calculates the probability of a trace coming from SoftmaxPolicy agents \
(who are not learning!)  given some cost function, values of cost weights and \
temperatures or a RandomPolicy agent.
It uses hyperopt.
"""
import json
from argparse import ArgumentParser
from pathlib import Path
import dill as pickle
import time

from more_itertools import powerset
import yaml
from cluster_utils import (
    get_args_from_yamls,
    get_human_trajectories,
    get_simulated_trajectories,
)
from costometer.agents.vanilla import SymmetricMouselabParticipant
from costometer.inference import HyperoptOptimizerInference, GridInference
from mouselab.cost_functions import *  # noqa
from mouselab.distributions import Categorical
from mouselab.graph_utils import get_structure_properties
from mouselab.policies import RandomPolicy, SoftmaxPolicy
from scipy import stats  # noqa
from get_myopic_voc_values import get_state_action_values
from hyperopt.early_stop import no_progress_loss

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
        "-d",
        "--pid",
        dest="pid",
        type=int,
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
        args = {**args, **get_args_from_yamls(
            args, attributes=["experiment_setting"]
        )
                }

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
        traces = get_human_trajectories(
            args["experiment"],
            pids=[inputs.pid],
            include_last_action=args["env_params"]["include_last_action"],
        )
        experiment_folder = args["experiment"]
        # data not simulated, no simulation params
        simulation_params = ""

    if inputs.prior_file is not None:
        yaml_path = str(
            path.joinpath(
                f"cluster/parameters/priors/{inputs.prior_file}.yaml"
            )
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

    cost_function = eval(args["cost_function"])
    if callable(eval(args["cost_function"])):
        cost_function_name = inputs.cost_function
    else:
        cost_function_name = None

    q_function_generator = lambda cost_parameters, a, g: get_state_action_values(
        experiment_setting=args["experiment_setting"],
        bmps_file=inputs.bmps_file,
        cost_function=cost_function,
        cost_parameters=cost_parameters,
        structure=structure_dicts,
        env_params=args["env_params"],
        alpha=a,
        gamma=g)

    all_results = {}
    for subset in powerset(args["constant_values"]):
        model_name = eval(args["model_name"])[subset]

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
            held_constant_cost_kwargs= {key : args["constant_values"][key] for key in subset},
            policy_parameters=prior_inputs["policy_parameters"],
            cost_function=cost_function,
            cost_parameters=prior_inputs["cost_parameters"],
            optimization_settings={"verbose":0,"max_evals":1000,"early_stop_fn": no_progress_loss(iteration_stop_count=500, percent_increase=0.0)},
        )

        softmax_opt_object.run()
        all_results[model_name + " with alpha, gamma"] = softmax_opt_object.get_optimization_results()

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
                "alpha": 1,
                "gamma": 1,
            },
            held_constant_cost_kwargs= {key : args["constant_values"][key] for key in subset},
            policy_parameters=prior_inputs["policy_parameters"],
            cost_function=cost_function,
            cost_parameters=prior_inputs["cost_parameters"],
            optimization_settings={"verbose": 0, "max_evals": 1000,
                                   "early_stop_fn": no_progress_loss(iteration_stop_count=500, percent_increase=0.0)})
        softmax_opt_object.run()
        all_results[model_name] = softmax_opt_object.get_optimization_results()

    random_opt_object = GridInference(
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
        cost_function=cost_function,
        cost_parameters={
            cost_parameter_arg: Categorical([cost_parameter_val], [1])
            for cost_parameter_arg, cost_parameter_val in args[
                "constant_values"
            ].items()
        },
        cost_function_name=cost_function_name,
    )

    random_opt_object.run()
    optimization_results = random_opt_object.get_output_df()
    all_results["Null"] = optimization_results

    # make experiment folder if it doesn't already exist
    path.joinpath(
        f"cluster/data/logliks_opt/{cost_function_name}/{experiment_folder}"
    ).mkdir(parents=True, exist_ok=True)

    filename = path.joinpath(
        f"cluster/data/logliks_opt/{cost_function_name}/{experiment_folder}/"
        f"SoftmaxPolicy_optimization_results"
        f"{simulation_params}_{inputs.pid}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    )
    with open(filename, "wb") as f:
        pickle.dump(all_results,f)
