"""Preprocess data downloaded as a CSV from our server
where URIs are not available for security reasons"""
import json
from argparse import ArgumentParser
from pathlib import Path

import dill as pickle
import pandas as pd
import rsa
import yaml
from download_tools.labeler import Labeler
from download_tools.save_participant_files import save_participant_files


def create_mturk_id_mapping(data_path):
    """ can be used to create labeler """
    pid_labeler = Labeler()
    with open(data_path.joinpath("mturk_id_mapping.pickle"), "wb") as f:
        pickle.dump(pid_labeler.labels, f)


def download_exp(experiment_name, data_path=None, save_path=None):
    if data_path is None:
        data_path = Path(__file__).parents[0]
    if save_path is None:
        save_path = data_path.joinpath("raw")

    # load inputs for experiment
    yaml_input = data_path.joinpath(f"inputs/yamls/experiments/{experiment_name}.yaml")
    with open(yaml_input, "r") as stream:
        kwargs = yaml.safe_load(stream)

    if "bonus_function" in kwargs:
        bonus_function = eval(kwargs["bonus_function"])
    else:
        bonus_function = None

    if "privatekey" in kwargs:
        # load private key
        with open(
            data_path.joinpath(f"inputs/keys/priv_{kwargs['privatekey']}"), "rb"
        ) as f:
            private_key = rsa.PrivateKey.load_pkcs1(f.read(), "PEM")
    else:
        private_key = None

    for session in kwargs["sessions"]:
        raw_df = pd.read_csv(data_path.joinpath(f"raw/csv/{session}.csv"))
        entries_with_data = raw_df[~pd.isnull(raw_df["datastring"])]

        # transform encrypted worker id > worker id
        if private_key:
            entries_with_data["workerid"] = entries_with_data["workerid"].apply(
                lambda workerid: rsa.decrypt(eval(workerid), private_key)
            )

        # now transform datastring to dictionary
        entries_with_data["datastring"] = entries_with_data["datastring"].apply(
            yaml.safe_load
        )
        # future preprocessing code will assume json format, not yaml
        entries_with_data["datastring"] = entries_with_data["datastring"].apply(
            json.dumps
        )
        participant_dicts = pd.DataFrame.to_dict(entries_with_data, orient="records")
        participant_dicts = [
            participant_dict
            for participant_dict in participant_dicts
            if len(participant_dict["workerid"]) > 10
            and len(participant_dict["workerid"]) < 30
        ]

        save_participant_files(
            participant_dicts,
            session,
            labeler=data_path.joinpath("mturk_id_mapping.pickle"),
            save_path=save_path,
            bonus_function=bonus_function,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp",
        dest="experiment_name",
        help="Experiment name to download (HIT ID file in ./hit_ids/HITS/)",
        metavar="experiment_name",
    )

    inputs = parser.parse_args()
    data_path = Path(__file__).resolve().parents[1]

    download_exp(inputs.experiment_name, data_path=data_path)
