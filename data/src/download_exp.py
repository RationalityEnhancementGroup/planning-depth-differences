from argparse import ArgumentParser
from pathlib import Path

import dill as pickle
import yaml
from download_tools.download_from_database import (
    download_from_database,
    load_database_uris,
)
from download_tools.labeler import Labeler
from download_tools.save_participant_files import save_participant_files


def create_mturk_id_mapping(data_path):
    pid_labeler = Labeler()
    with open(data_path.joinpath("/mturk_id_mapping.pickle"), "wb") as f:
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

    # load database URIs
    load_database_uris(data_path)

    example_participant_dicts = download_from_database(
        data_path.joinpath(f"hit_ids/{experiment_name}.txt"), kwargs["database_key"]
    )
    save_participant_files(
        example_participant_dicts,
        experiment_name,
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

    args = parser.parse_args()
    data_path = Path(__file__).resolve().parents[1]

    if args.experiment_name:
        download_exp(args.experiment_name, data_path=data_path)
    else:
        for hit in data_path.glob("hit_ids/*.txt"):
            experiment_name = hit.stem
            download_exp(experiment_name, data_path=data_path)
