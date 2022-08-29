from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import yaml
from costometer.utils import get_param_string

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

    inputs = parser.parse_args()

    path = Path(__file__).resolve().parents[3]

    yaml_path = path.joinpath(
        f"data/inputs/yamls/cost_functions/{inputs.cost_function}.yaml"
    )
    with open(str(yaml_path), "r") as stream:
        args = yaml.safe_load(stream)

    if inputs.cost_parameter_values == "*":
        cost_string = "*"
        save_cost_string = "all"
    else:
        cost_parameter_dict = {
            cost_parameter_arg: float(arg)
            for arg, cost_parameter_arg in zip(
                inputs.cost_parameter_values.split(","), args["cost_parameter_args"]
            )
        }

        cost_string = get_param_string(cost_parameter_dict)
        save_cost_string = cost_string

    full_df = pd.concat(
        [
            pd.read_csv(f, index_col=0)
            for f in path.glob(
                f"cluster/data/logliks/{inputs.cost_function}/{inputs.experiment}"
                f"/MCL_optimization_results_{cost_string}_{inputs.model_yaml}"
                f"_{inputs.feature_yaml}_{inputs.constant_yaml}*.csv"
            )
        ]
    )

    full_df.reset_index(drop=True).to_feather(
        path.joinpath(
            f"cluster/data/logliks/{inputs.cost_function}/"
            f"MCL_{inputs.experiment}_{save_cost_string}"
            f"_{inputs.model_yaml}_{inputs.feature_yaml}"
            f"_{inputs.constant_yaml}.feather"
        )
    )
