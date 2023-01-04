"""
This scripts calculates the probability of a trace coming from SymmetricMCLParticipant \
agents (e.g. they are learning) given some cost function, values of cost weights and \
prior features weights.
"""
import json
import time
from argparse import ArgumentParser
from pathlib import Path

import dill as pickle
import yaml
from cluster_utils import (
    get_args_from_yamls,
    get_human_trajectories,
    get_simulated_trajectories,
)
from costometer.agents import SymmetricMCLParticipant
from costometer.inference import MCLInference
from costometer.utils import get_param_string
from hyperopt.early_stop import no_progress_loss
from mcl_toolbox.utils.feature_normalization import get_new_feature_normalization
from mouselab.cost_functions import *  # noqa
from mouselab.distributions import Categorical
from mouselab.graph_utils import get_structure_properties
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
        "-m",
        "--model-yaml",
        dest="model_yaml",
        help="Model YAML file",
        type=str,
    )
    parser.add_argument(
        "-f",
        "--feature-yaml",
        dest="feature_yaml",
        help="Feature YAML file",
        type=str,
    )
    parser.add_argument(
        "-p",
        "--prior-json",
        dest="prior_json",
        help="File with priors for MCL features",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--constant-yaml",
        dest="constant_yaml",
        help="Constant YAML",
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
        "-v",
        "--values",
        dest="cost_parameter_values",
        help="Cost parameter values as comma separated string, e.g. '1.00,2.00'",
        type=str,
    )
    parser.add_argument(
        "-d",
        "--pid",
        dest="pid",
        help="pid",
        type=int,
        default=None,
    )

    inputs = parser.parse_args()

    if "*" in inputs.experiment:
        args = get_args_from_yamls(vars(inputs), attributes=["cost_function"])
        args["experiment"] = inputs.experiment
        args["experiment_setting"] = inputs.experiment.split("/")[0]
    else:
        args = get_args_from_yamls(
            vars(inputs), attributes=["cost_function", "experiment"]
        )

    path = Path(__file__).resolve().parents[2]

    # if wild card in experiment name, this is file pattern for simulated trajectories
    if "*" in args["experiment"]:
        # TODO pid here
        traces = get_simulated_trajectories(args["experiment"])
        experiment_folder = "simulated/" + "/".join(args["experiment"].split("/")[:-1])
    else:
        traces = get_human_trajectories(
            args["experiment"],
            pids=[inputs.pid] if inputs.pid else inputs.pid,
            include_last_action=args["env_params"]["include_last_action"],
        )
        experiment_folder = args["experiment"]

    # load yaml inputs
    model_attributes_file = path.joinpath(
        f"cluster/parameters/mcl/run/{inputs.model_yaml}.yaml"
    )
    with open(str(model_attributes_file), "r") as stream:
        model_attributes = yaml.safe_load(stream)

    feature_file = path.joinpath(
        f"cluster/parameters/mcl/features/{inputs.feature_yaml}.yaml"
    )
    with open(str(feature_file), "r") as stream:
        features = yaml.safe_load(stream)

    if inputs.constant_yaml:
        constant_file = path.joinpath(
            f"cluster/parameters/mcl/constant/{inputs.constant_yaml}.yaml"
        )
        with open(str(constant_file), "r") as stream:
            held_constant = yaml.safe_load(stream)
    else:
        held_constant = None

    participant_kwargs = {
        "model_attributes": model_attributes,
        "features": features["features"],
    }

    # load normalized features
    normalized_file = path.joinpath(
        f"cluster/parameters/mcl/normalized/{inputs.feature_yaml}.pkl"
    )
    if not normalized_file.is_file():
        new_normalized = get_new_feature_normalization(
            participant_kwargs["features"],
            exp_setting=args["experiment_setting"],
            num_trials=30,
            num_simulations=50,
        )
        with open(
            str(normalized_file),
            "wb",
        ) as f:
            pickle.dump(new_normalized, f)
        participant_kwargs["normalized_features"] = new_normalized
    else:
        with open(
            str(normalized_file),
            "rb",
        ) as f:
            normalized_features = pickle.load(f)
        participant_kwargs["normalized_features"] = normalized_features

    if inputs.prior_json is not None:
        json_path = path.joinpath(
            f"cluster/parameters/mcl/priors/{inputs.prior_json}.json"
        )
        with open(str(json_path), "r") as f:
            priors = json.load(f)
    else:
        priors = None

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

    # make experiment folder if it doesn't already exist
    path.joinpath(
        f"cluster/data/logliks/{args['cost_function']}/{experiment_folder}"
    ).mkdir(parents=True, exist_ok=True)

    if "structure" in args:
        with open(
            Path(__file__)
            .parents[1]
            .joinpath(f"data/exp_inputs/structure/{args['structure']}.json"),
            "rb",
        ) as f:
            structure_data = json.load(f)

        structure_dicts = get_structure_properties(structure_data)

        # for cost functions that need graph properties
        participant_kwargs["additional_mouselab_kwargs"] = {
            "mdp_graph_properties": structure_dicts
        }
    else:
        structure_dicts = None

    cost_function = eval(args["cost_function"])

    mcl_ray_object = MCLInference(
        traces=traces,
        participant_class=SymmetricMCLParticipant,
        participant_kwargs=participant_kwargs,
        held_constant_policy_kwargs=held_constant,
        cost_function=cost_function,
        cost_parameters=cost_parameters,
        optimization_settings={
            "max_evals": 1000,
            "early_stop_fn": no_progress_loss(
                iteration_stop_count=200, percent_increase=0.0
            ),
        },
    )

    mcl_ray_object.run()

    optimization_results = mcl_ray_object.get_optimization_results()
    if inputs.pid is not None:
        filename = path.joinpath(
            f"cluster/data/logliks/{cost_function.__name__}/{experiment_folder}/"
            f"MCL_optimization_results_{get_param_string(cost_parameter_dict)}"
            f"_{inputs.model_yaml}_{inputs.feature_yaml}_"
            f"{inputs.constant_yaml}_{inputs.pid}_{time.strftime('%Y%m%d-%H%M')}.csv"
        )
    else:
        filename = path.joinpath(
            f"cluster/data/logliks/{cost_function.__name__}/{experiment_folder}/"
            f"MCL_optimization_results_{get_param_string(cost_parameter_dict)}"
            f"+{inputs.model_yaml}_{inputs.feature_yaml}_"
            f"{inputs.constant_yaml}_{time.strftime('%Y%m%d-%H%M')}.csv"
        )
    optimization_results.to_csv(filename)
