import logging
import sys
from pathlib import Path

import numpy as np
import pingouin as pg
from costometer.utils import get_correlation_text, get_mann_whitney_text
from costometer.utils.scripting_utils import standard_parse_args

if __name__ == "__main__":
    irl_path = Path(__file__).resolve().parents[4]
    analysis_obj, inputs, subdirectory = standard_parse_args(
        description=sys.modules[__name__].__doc__,
        irl_path=irl_path,
        filename=Path(__file__).stem,
        default_experiment="SoftmaxRecovery",
    )

    optimization_data = analysis_obj.query_optimization_data(
        excluded_parameters=analysis_obj.analysis_details.excluded_parameters
    )

    model_params = set(analysis_obj.cost_details.constant_values) - set(
        analysis_obj.analysis_details.excluded_parameters
    )

    hdi_ranges = analysis_obj.load_hdi_ranges(
        excluded_parameter_str=analysis_obj.analysis_details.excluded_parameter_str
    )

    for parameter in model_params:
        optimization_data[f"min_{parameter}"] = optimization_data["trace_pid"].apply(
            lambda pid: hdi_ranges[pid][parameter][0]
        )
        optimization_data[f"max_{parameter}"] = optimization_data["trace_pid"].apply(
            lambda pid: hdi_ranges[pid][parameter][1]
        )

    for parameter in model_params:
        optimization_data[f"{parameter}_in"] = optimization_data.apply(
            lambda row: (row[f"sim_{parameter}"] <= row[f"max_{parameter}"])
            and (row[f"sim_{parameter}"] >= row[f"min_{parameter}"]),
            axis=1,
        )
        optimization_data[f"{parameter}_spread"] = optimization_data.apply(
            lambda row: row[f"max_{parameter}"] - row[f"min_{parameter}"], axis=1
        )

    logging.info("Statistics for spread of parameters")
    for parameter in model_params:
        logging.info(
            f"{analysis_obj.cost_details.latex_mapping[parameter]}"
            f" & ${optimization_data[f'{parameter}_spread'].mean():.2f}$"
            f" (${optimization_data[f'{parameter}_spread'].std():.2f}$)"
        )

    # for cases where we don't vary temperature
    for parameter in model_params:
        logging.info("----------")
        logging.info(f"Correlation between spread of {parameter} and temperature")
        logging.info("----------")
        correlation_object = pg.corr(
            optimization_data["sim_temp"], optimization_data[f"{parameter}_spread"]
        )
        logging.info(get_correlation_text(correlation_object))

    for parameter in model_params:
        logging.info("----------")
        logging.info(f"Amount of time true {parameter} is in the outputted interval")
        logging.info("----------")
        logging.info(f"{optimization_data[f'{parameter}_in'].mean():.2f}")

    for parameter in model_params:
        optimization_data[f"diff_{parameter}"] = optimization_data.apply(
            lambda row: np.sqrt((row[parameter] - row[f"sim_{parameter}"]) ** 2),
            axis=1,
        )

        if len(optimization_data[f"{parameter}_in"].unique()) == 1:
            logging.info("----------")
            logging.info(
                f"True {parameter} value is always in or outside of outputted interval:"
            )
            logging.info(optimization_data[f"{parameter}_in"].unique())
        else:
            logging.info("----------")
            logging.info(
                f"Difference between error when true {parameter} "
                f"parameter value is in outputted interval vs not."
            )
            logging.info("----------")
            comparison = pg.mwu(
                optimization_data[optimization_data[f"{parameter}_in"]][
                    f"diff_{parameter}"
                ],
                optimization_data[~optimization_data[f"{parameter}_in"]][
                    f"diff_{parameter}"
                ],
            )
            logging.info(get_mann_whitney_text(comparison))

    logging.info("----------")
    logging.info("Correlation between error in MAP estimate and spread")
    logging.info("----------")
    for parameter in model_params:
        correlation_object = pg.corr(
            optimization_data[f"{parameter}_spread"],
            optimization_data[f"diff_{parameter}"],
        )
        logging.info(
            f"{analysis_obj.cost_details.latex_mapping[parameter]}"
            f" & {get_correlation_text(correlation_object, table=True)}"
        )

    logging.info("----------")
    logging.info("Correlation between MAP estimate and spread")
    logging.info("----------")
    for parameter in model_params:
        correlation_object = pg.corr(
            optimization_data[parameter],
            optimization_data[f"{parameter}_spread"],
        )
        logging.info(
            f"{analysis_obj.cost_details.latex_mapping[parameter]}"
            f" & {get_correlation_text(correlation_object, table=True)}"
        )

    logging.info("----------")
    logging.info("Correlation between true value and spread")
    logging.info("----------")
    for parameter in model_params:
        correlation_object = pg.corr(
            optimization_data[f"sim_{parameter}"],
            optimization_data[f"{parameter}_spread"],
        )
        logging.info(
            f"{analysis_obj.cost_details.latex_mapping[parameter]}"
            f" & {get_correlation_text(correlation_object, table=True)}"
        )
