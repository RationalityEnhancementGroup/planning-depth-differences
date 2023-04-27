from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import pingouin as pg
from costometer.utils import AnalysisObject, get_correlation_text, set_font_sizes

set_font_sizes()


def plot_hdi(hdi_range_dict, param):
    plotting_dict = {
        num: plotting_info
        for num, plotting_info in zip(
            range(len(hdi_range_dict)),
            sorted(
                [hdi_range[param] for pid, hdi_range in hdi_range_dict.items()],
                key=lambda item: item,
            ),
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
        default="methods/static",
        dest="experiment_subdirectory",
        metavar="experiment_subdirectory",
    )
    inputs = parser.parse_args()

    irl_path = Path(__file__).resolve().parents[4]
    data_path = irl_path.joinpath(f"analysis/{inputs.experiment_subdirectory}")

    analysis_obj = AnalysisObject(
        inputs.experiment_name,
        irl_path=irl_path,
        experiment_subdirectory=inputs.experiment_subdirectory,
    )
    optimization_data = analysis_obj.query_optimization_data(
        excluded_parameters=analysis_obj.excluded_parameters
    )
    hdi_ranges = analysis_obj.load_hdi_ranges(analysis_obj.excluded_parameters)

    for parameter in analysis_obj.cost_details["constant_values"]:
        print("==========")

        plot_hdi(hdi_ranges, parameter)
        plt.savefig(
            data_path.joinpath(f"figs/{inputs.experiment_name}_{parameter}_hdi.png"),
            bbox_inches="tight",
        )

        optimization_data[f"{parameter}_spread"] = optimization_data["trace_pid"].apply(
            lambda pid: hdi_ranges[pid][parameter][1] - hdi_ranges[pid][parameter][0]
        )

        print("----------")
        print(f"Correlation between BIC and spread for {parameter}")
        print("----------")
        correlation_object = pg.corr(
            optimization_data["bic"], optimization_data[f"{parameter}_spread"]
        )
        print(parameter)
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

    optimization_data
