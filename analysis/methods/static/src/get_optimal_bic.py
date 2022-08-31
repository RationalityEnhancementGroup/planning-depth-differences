from argparse import ArgumentParser
from pathlib import Path

import dill as pickle
import matplotlib.pyplot as plt
import numpy as np
import pingouin as pg
import seaborn as sns
from costometer.utils import AnalysisObject, get_correlation_text, set_font_sizes
from scipy import stats  # noqa

set_font_sizes()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--cost-function",
        dest="cost_function",
        default="linear_depth",
        type=str,
    )
    parser.add_argument(
        "-e",
        "--exp",
        dest="experiment_name",
    )
    parser.add_argument(
        "-s",
        "--subdirectory",
        dest="experiment_subdirectory",
        metavar="experiment_subdirectory",
    )
    parser.add_argument(
        "-n",
        "--num-subjects",
        default=130,
        dest="num_subjects",
    )
    inputs = parser.parse_args()

    data_path = Path(__file__).resolve().parents[1]
    irl_path = Path(__file__).resolve().parents[4]

    analysis_obj = AnalysisObject(
        inputs.experiment_name,
        irl_path=irl_path,
        experiment_subdirectory=inputs.experiment_subdirectory,
    )

    data_path.joinpath("log").mkdir(parents=True, exist_ok=True)
    data_path.joinpath("data").mkdir(parents=True, exist_ok=True)

    optimization_data = analysis_obj.query_optimization_data()
    optimization_data = optimization_data[
        optimization_data["Model Name"] == "Effort Cost and Planning Depth"
    ]

    sim_cols = [col for col in list(optimization_data) if "sim_" in col]
    mean_over_sim_param = optimization_data.groupby(sim_cols).mean().reset_index()
    mean_over_sim_param = mean_over_sim_param.rename(columns={"bic": "BIC"})

    cost_details = analysis_obj.cost_details[analysis_obj.cost_functions[0]]

    # rename parameters to pretty names
    mean_over_sim_param = mean_over_sim_param.rename(
        columns=dict(
            zip(
                [
                    f"sim_{cost_param}"
                    for cost_param in cost_details["cost_parameter_args"]
                ],
                cost_details["cost_parameter_names"],
            )
        )
    )

    pivoted_mean_ll = mean_over_sim_param.pivot(
        index=cost_details["cost_parameter_names"][0],
        columns=cost_details["cost_parameter_names"][1],
        values="BIC",
    )

    sns.heatmap(pivoted_mean_ll)
    plt.savefig(
        data_path.joinpath(f"figs/{inputs.experiment_name}.png"),
        bbox_inches="tight",
    )

    simulated_means = [
        optimization_data[optimization_data["trace_pid"] == num_simulation]
        .sample(inputs.num_subjects)["bic"]
        .sum()
        for num_simulation in range(200)
    ]
    intended_means = [
        optimization_data[
            optimization_data.apply(
                lambda row: np.all(
                    [
                        row[cost_param] == cost_val
                        for cost_param, cost_val in cost_details[
                            "constant_values"
                        ].items()
                    ]
                ),
                axis=1,
            )
        ]
        .sample(inputs.num_subjects)["bic"]
        .sum()
        for _ in range(10)
    ]
    with open(data_path.joinpath(f"data/{inputs.experiment_name}.pickle"), "wb") as f:
        pickle.dump({"all": simulated_means, "intended": intended_means}, f)

    for param in analysis_obj.cost_details[inputs.cost_function][
        "cost_parameter_args"
    ] + ["temp"]:
        print("----------")
        print(f"Correlation between {param} and BIC")
        print("----------")
        correlation = pg.corr(mean_over_sim_param[param], mean_over_sim_param["BIC"])
        print(get_correlation_text(correlation))
