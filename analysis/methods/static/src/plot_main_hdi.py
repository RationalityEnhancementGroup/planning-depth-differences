import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pingouin as pg
from costometer.utils import get_correlation_text
from costometer.utils.scripting_utils import standard_parse_args


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
    irl_path = Path(__file__).resolve().parents[4]
    analysis_obj, inputs, subdirectory = standard_parse_args(
        description=sys.modules[__name__].__doc__,
        irl_path=irl_path,
        filename=Path(__file__).stem,
    )

    optimization_data = analysis_obj.query_optimization_data(
        excluded_parameters=analysis_obj.analysis_details.excluded_parameters
    )
    hdi_ranges = analysis_obj.load_hdi_ranges(
        analysis_obj.analysis_details.excluded_parameter_str
    )

    for parameter in analysis_obj.cost_details.constant_values:
        logging.info("==========")

        plot_hdi(hdi_ranges, parameter)
        plt.savefig(
            subdirectory.joinpath(f"figs/{inputs.experiment_name}_{parameter}_hdi.png"),
            bbox_inches="tight",
        )

        optimization_data[f"{parameter}_spread"] = optimization_data["trace_pid"].apply(
            lambda pid: hdi_ranges[pid][parameter][1] - hdi_ranges[pid][parameter][0]
        )

        logging.info("----------")
        logging.info(f"Correlation between BIC and spread for {parameter}")
        logging.info("----------")
        correlation_object = pg.corr(
            optimization_data["bic"], optimization_data[f"{parameter}_spread"]
        )
        logging.info(parameter)
        logging.info(get_correlation_text(correlation_object))

        logging.info("----------")
        logging.info(f"Correlation between BIC and MAP parameter value for {parameter}")
        logging.info("----------")
        correlation_object = pg.corr(
            optimization_data["bic"], optimization_data[f"{parameter}"]
        )
        logging.info(get_correlation_text(correlation_object))

        logging.info("----------")
        logging.info(
            f"Correlation between parameter spread and "
            f"MAP parameter value for {parameter}"
        )
        logging.info("----------")
        correlation_object = pg.corr(
            optimization_data[f"{parameter}_spread"],
            optimization_data[f"{parameter}"],
        )
        logging.info(get_correlation_text(correlation_object))

    optimization_data
