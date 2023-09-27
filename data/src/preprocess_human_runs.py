from argparse import ArgumentParser
from pathlib import Path

import papermill as pm
import yaml


def preprocess_human_data(yaml_file):
    yaml_path = Path(yaml_file).resolve()
    with open(str(yaml_path), "r") as stream:
        inputs = yaml.safe_load(stream)

    # add experiment setting arguments
    with open(
        str(
            yaml_path.parents[1].joinpath(
                f"experiment_settings/{inputs['experiment_setting']}.yaml"
            )
        ),
        "r",
    ) as stream:
        inputs = {**inputs, **yaml.safe_load(stream)}

    inputs["analysis_run"] = yaml_path.stem
    # posix path doesn't work in the parameters dictionary
    inputs["data_path"] = str(yaml_path.parents[3])

    output_file = Path(inputs["data_path"]).joinpath(
        f"Preprocessing_{inputs['analysis_run']}.ipynb"
    )

    if "final_quest" in inputs and inputs["final_quest"]:
        input_file = (
            Path(__file__)
            .parents[1]
            .joinpath("templates/preprocessing_human_quest_main.ipynb")
        )
    elif "column_mapping" in inputs:
        input_file = (
            Path(__file__)
            .parents[1]
            .joinpath("templates/preprocessing_human_shared_data.ipynb")
        )
    else:
        input_file = (
            Path(__file__).parents[1].joinpath("templates/preprocessing_human.ipynb")
        )

        # return
    pm.execute_notebook(
        input_path=str(input_file),
        output_path=str(output_file),
        kernel_name="planning-depth-differences",
        parameters=inputs,
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
        yaml_file = str(
            data_path.joinpath(f"inputs/yamls/experiments/{args.experiment_name}.yaml")
        )
        preprocess_human_data(yaml_file)
    else:
        for hit in data_path.glob("hit_ids/*.txt"):
            experiment_name = hit.stem

            yaml_file = str(
                data_path.joinpath(f"inputs/yamls/experiments/{experiment_name}.yaml")
            )
            preprocess_human_data(yaml_file)
