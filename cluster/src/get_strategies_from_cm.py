from argparse import ArgumentParser
from pathlib import Path

import dill as pickle
import pandas as pd
import yaml
from mcl_toolbox.computational_microscope.computational_microscope import (
    ComputationalMicroscope,
)
from mcl_toolbox.global_vars import features, strategies
from mcl_toolbox.utils.experiment_utils import Experiment
from mcl_toolbox.utils.feature_normalization import get_new_feature_normalization
from mcl_toolbox.utils.learning_utils import (
    construct_repeated_pipeline,
    create_mcrl_reward_distribution,
    get_modified_weights,
)
from mouselab.envs.registry import registry

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--experiment",
        dest="experiment",
        help="Experiment",
        type=str,
    )

    inputs = parser.parse_args()
    path = Path(__file__).resolve().parents[2]

    # read in experiment file
    yaml_path = path.joinpath(f"data/inputs/yamls/experiments/{inputs.experiment}.yaml")
    with open(yaml_path, "r") as stream:
        experiment_details = yaml.safe_load(stream)

    # read in experiment setting variables
    yaml_path = path.joinpath(
        f"data/inputs/yamls/experiment_settings/"
        f"{experiment_details['experiment_setting']}.yaml"
    )
    with open(yaml_path, "r") as stream:
        experiment_details = {**experiment_details, **yaml.safe_load(stream)}

    mouselab_data = pd.read_csv(
        path.joinpath(f"data/processed/{inputs.experiment}/mouselab-mdp.csv")
    )
    num_trials = mouselab_data.groupby(["pid"]).count()["trial_index"].max()

    reward_distributions = create_mcrl_reward_distribution(
        experiment_details["experiment_setting"]
    )
    repeated_pipeline = construct_repeated_pipeline(
        registry(experiment_details["experiment_setting"]).branching,
        reward_distributions,
        num_trials,
    )

    strategy_space = strategies.strategy_space
    microscope_features = features.microscope
    strategy_weights = strategies.strategy_weights

    # load normalized features
    normalized_file = path.joinpath(
        f"cluster/parameters/mcl/normalized/"
        f"microscope_{experiment_details['experiment_setting']}.pkl"
    )
    if not normalized_file.is_file():
        normalized_features = get_new_feature_normalization(
            microscope_features,
            exp_setting=experiment_details["experiment_setting"],
            num_trials=num_trials,
            num_simulations=50,
        )
        with open(
            str(normalized_file),
            "wb",
        ) as f:
            pickle.dump(normalized_features, f)
    else:
        with open(
            str(normalized_file),
            "rb",
        ) as f:
            normalized_features = pickle.load(f)

    W = get_modified_weights(strategy_space, strategy_weights)
    cm = ComputationalMicroscope(
        repeated_pipeline,
        strategy_space,
        W,
        microscope_features,
        normalized_features=normalized_features,
    )

    exp = Experiment(
        inputs.experiment,
        cm=cm,
        data_path=path.joinpath("data/processed"),
    )

    exp.infer_strategies()

    path.joinpath("cluster/data/cm/").mkdir(parents=True, exist_ok=True)
    with open(path.joinpath(f"cluster/data/cm/{inputs.experiment}.pkl"), "wb") as f:
        pickle.dump(exp, f)
