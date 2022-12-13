from argparse import ArgumentParser
from pathlib import Path

import dill as pickle
import matplotlib.pyplot as plt
import pingouin as pg
from costometer.utils import AnalysisObject, get_correlation_text, set_font_sizes

set_font_sizes()


def plot_hdi(hdi_range_dict, param):
    plotting_dict = {
        num: plotting_info
        for num, plotting_info in zip(
            range(len(hdi_range_dict["test"][param])),
            sorted(hdi_range_dict["test"][param].values(), key=lambda item: item),
        )
    }
    plt.figure(figsize=(12, 8), dpi=80)
    ax = plt.gca()

    xs = []
    ys = []
    for y, plotting_info in plotting_dict.items():
        if plotting_info[0] == plotting_info[1]:
            xs.append(plotting_info[0])
            ys.append(y)
        ax.hlines(
            y=y,
            xmin=plotting_info[0],
            xmax=plotting_info[1],
            linewidth=2,
            color="g",
        )
    plt.plot(xs, ys, marker="o", markersize=6, color="b", linestyle="None")

    plt.title(f"{param.title()}")
    plt.xlabel("Value")
    plt.ylabel("Participants")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp",
        dest="experiment_name",
        metavar="experiment_name",
    )
    parser.add_argument(
        "-s",
        "--subdirectory",
        dest="experiment_subdirectory",
        metavar="experiment_subdirectory",
    )
    parser.add_argument(
        "-c",
        "--cost-function",
        dest="cost_function",
        type=str,
        default="dist_depth_forw",
    )
    inputs = parser.parse_args()

    data_path = Path(__file__).resolve().parents[1]
    irl_path = Path(__file__).resolve().parents[4]

    analysis_obj = AnalysisObject(
        inputs.experiment_name,
        irl_path=irl_path,
        experiment_subdirectory=inputs.experiment_subdirectory,
    )
    optimization_data = analysis_obj.query_optimization_data()
    optimization_data = optimization_data[
        optimization_data["Model Name"] == "Effort Cost and Planning Depth"
    ]

    hdi_file = data_path.joinpath(
        f"data/{inputs.experiment_name}/"
        f"{inputs.experiment_name}_{inputs.cost_function}_hdi.pickle"
    )
    with open(
        hdi_file,
        "rb",
    ) as f:
        hdi_ranges = pickle.load(f)

    for parameter in hdi_ranges["test"].keys():
        print("==========")

        plot_hdi(hdi_ranges, parameter)
        plt.savefig(
            data_path.joinpath(f"figs/{inputs.experiment_name}_{parameter}_hdi.png"),
            bbox_inches="tight",
        )

        optimization_data[f"{parameter}_spread"] = optimization_data["trace_pid"].apply(
            lambda pid: hdi_ranges["test"][parameter][pid][1]
            - hdi_ranges["test"][parameter][pid][0]
        )

        print("----------")
        print(f"Correlation between BIC and spread for {parameter}")
        print("----------")
        correlation_object = pg.corr(
            optimization_data["bic"], optimization_data[f"{parameter}_spread"]
        )
        print(get_correlation_text(correlation_object))

        print("----------")
        print(f"Correlation between BIC and MAP parameter value for {parameter}")
        print("----------")
        correlation_object = pg.corr(
            optimization_data["bic"], optimization_data[f"{parameter}"]
        )
        print(get_correlation_text(correlation_object))

        print("----------")
        print(
            f"Correlation between parameter spread and "
            f"MAP parameter value for {parameter}"
        )
        print("----------")
        correlation_object = pg.corr(
            optimization_data[f"{parameter}_spread"],
            optimization_data[f"{parameter}"],
        )
        print(get_correlation_text(correlation_object))
