"""Utilities for running some of the cluster scripts in this folder"""
import os
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd
import yaml
from costometer.utils import (
    get_states_for_trace,
    get_trajectories_from_participant_data,
)
from mouselab.distributions import Categorical
from mouselab.envs.registry import register
from mouselab.envs.reward_settings import high_increasing_reward


def create_test_env(name) -> None:
    """
    Register a (given) test environment for unit tests
    :return: None, registers test env
    """
    if name == "small_test_case":
        register(
            name=name,
            branching=[1, 2],
            reward_inputs="depth",
            reward_dictionary={1: Categorical([-500]), 2: Categorical([-60, 60])},
        )
    elif name == "cogsci_learning":
        register(
            name=name,
            branching=[3, 1, 2],
            reward_inputs="depth",
            reward_dictionary={
                1: Categorical([-8, -4, 4, 8]),  # usually ([-4, -2, 2, 4]),
                2: Categorical([-8, -4, 4, 8]),
                3: Categorical([-48, -24, 24, 48]),
            },
        )
    elif name == "mini_variance":
        register(
            name=name,
            branching=[3, 1, 2],
            reward_inputs="depth",
            reward_dictionary={
                1: Categorical([-2, -1, 1, 2]),  # usually ([-4, -2, 2, 4]),
                2: Categorical([-8, -4, 4, 8]),
                3: Categorical([-48, -24, 24, 48]),
            },
        )
    elif name == "zero_variance":
        register(
            name=name,
            branching=[3, 1, 2],
            reward_inputs="depth",
            reward_dictionary={
                1: Categorical([1, 1, 1, 1]),  # usually ([-4, -2, 2, 4]),
                2: Categorical([-8, -4, 4, 8]),
                3: Categorical([-48, -24, 24, 48]),
            },
        )
    elif name == "large_variance":
        register(
            name=name,
            branching=[3, 1, 2],
            reward_inputs="depth",
            reward_dictionary={
                1: Categorical([-48, -24, 24, 48]),  # usually ([-4, -2, 2, 4]),
                2: Categorical([-8, -4, 4, 8]),
                3: Categorical([-48, -24, 24, 48]),
            },
        )
    elif name == "reduced_variance":
        register(
            name=name,
            branching=[3, 1, 2],
            reward_inputs="depth",
            reward_dictionary={
                1: Categorical([-4, 4]),
                2: Categorical([-8, 8]),
                3: Categorical([-48, 48]),
            },
        )
    elif name == "reduced_root":
        register(
            name=name,
            branching=[2, 1, 2],
            reward_inputs="depth",
            reward_dictionary=high_increasing_reward,
        )
    elif name == "reduced_leaf":
        register(
            name=name,
            branching=[3, 1, 1],
            reward_inputs="depth",
            reward_dictionary=high_increasing_reward,
        )
    elif name == "reduced_middle":
        register(
            name=name,
            branching=[3, 2],
            reward_inputs="depth",
            reward_dictionary={
                1: Categorical([-4, -2, 2, 4]),
                2: Categorical([-48, -24, 24, 48]),
            },
        )


def get_human_trajectories(
    exp_name: str,
    data_path: Union[str, bytes, os.PathLike] = None,
    pids: List[int] = None,
    include_last_action: bool = False,
) -> List[Dict[str, List]]:
    """
    Get human trajectories from experiment in data folder
    :param exp_name: name experiment is saved under
    :param data_path: path where data is saved, including YAMLs with \
    experiment information
    :param pids:
    :param include_last_action:
    :return: traces for all participants
    """
    if not data_path:
        data_path = Path(__file__).resolve().parents[2].joinpath("data")

    yaml_path = data_path.joinpath(f"inputs/yamls/experiments/{exp_name}.yaml")
    with open(str(yaml_path), "r") as stream:
        experiment_setting = yaml.safe_load(stream)["experiment_setting"]

    mouselab_file = data_path.joinpath(f"processed/{exp_name}/mouselab-mdp.csv")
    mouselab_data = pd.read_csv(mouselab_file)

    if pids is None:
        pids = mouselab_data["pid"].unique()

    traces = get_trajectories_from_participant_data(
        mouselab_data[mouselab_data["pid"].isin(pids)],
        experiment_setting=experiment_setting,
        include_last_action=include_last_action,
    )

    return traces


def get_simulated_trajectories(
    file_pattern: str,
    experiment_setting: str,
    simulated_trajectory_path: Union[str, bytes, os.PathLike] = None,
    additional_mouselab_kwargs: dict = None,
) -> List[Dict[str, List]]:
    """
    Given path to simulated trajectories and a file pattern, outputs traces \
    to be used in inference
    :param file_pattern: corresponding to the type of simulated trajectory \
    we're interested in, as a glob partner
    :param simulated_trajectory_path: where the simulated trajectories are located
    :return:
    """
    if simulated_trajectory_path is None:
        simulated_trajectory_path = (
            Path(__file__).resolve().parents[1].joinpath("data/trajectories")
        )

    if "*" in file_pattern:
        files = simulated_trajectory_path.glob(file_pattern)
    else:
        files = [file_pattern]

    mouselab_data = pd.concat(
        [pd.read_csv(sim_file, index_col=0) for sim_file in files]
    )

    mouselab_data = mouselab_data.fillna(value="None")
    sim_cols = [col for col in list(mouselab_data) if "sim_" in col]

    imploded_by_episode_df = (
        mouselab_data.groupby(["pid", "i_episode"] + sim_cols)
        .agg(lambda x: x.tolist())
        .reset_index()
    )

    if "ground_truth" in list(imploded_by_episode_df):
        imploded_by_episode_df["ground_truth"] = imploded_by_episode_df[
            "ground_truth"
        ].apply(lambda ground_truths: eval(ground_truths[0]))
    else:
        imploded_by_episode_df["ground_truth"] = None

    state_info = imploded_by_episode_df.apply(
        lambda row: get_states_for_trace(
            row["actions"],
            experiment_setting,
            ground_truth=row["ground_truth"],
            **additional_mouselab_kwargs,
        ),
        axis=1,
    ).to_dict()
    state_info_dict = pd.DataFrame.from_dict(state_info).transpose()

    for param in list(state_info_dict):
        imploded_by_episode_df[param] = state_info_dict[param]

    imploded_df = (
        imploded_by_episode_df.groupby(["pid"] + sim_cols)
        .agg(lambda x: x.tolist())
        .reset_index()
    )
    traces = imploded_df.to_dict("records")

    for trace in traces:
        trace["pid"] = [trace["pid"]] * len(trace["i_episode"])

    return traces


def get_args_from_yamls(
    input_dictionary: Dict[Any, Any],
    attributes: List[str],
    yaml_path: Union[str, bytes, os.PathLike] = None,
) -> Dict[Any, Any]:
    """
    Loads and combined all information from input YAMLs into one dictionary
    :param input_dictionary: dictionary of input : input variable, for Namespace \
    object from ArgParse ars(obj) will give you this
    :param attributes: attributes expected to be in Namespace object
    :param yaml_path: path where yamls are saved
    :return: dictionary of all settings to be used, useful for cluster scripts
    """
    if yaml_path is None:
        yaml_path = Path(__file__).resolve().parents[2].joinpath("data/inputs/yamls")

    path_to_yamls = [
        yaml_path.joinpath(f"{attribute}s/{input_dictionary[attribute]}.yaml")
        for attribute in attributes
        if input_dictionary[attribute] is not None
    ]
    args = {
        attribute: input_dictionary[attribute]
        for attribute in attributes
        if input_dictionary[attribute] is not None
    }
    for yaml_path in path_to_yamls:
        with open(str(yaml_path), "r") as stream:
            args = {**args, **yaml.safe_load(stream)}
    return args
