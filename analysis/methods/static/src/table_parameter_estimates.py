import logging
from argparse import ArgumentParser
from itertools import combinations
from pathlib import Path

import numpy as np
import pingouin as pg
from costometer.utils import (
    AnalysisObject,
    get_mann_whitney_text,
    set_plotting_and_logging_defaults,
)

if __name__ == "__main__":
    """
    Example usage:
    python src/table_parameter_estimates.py
    """
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp",
        dest="experiment_names",
        default=["MainExperiment", "c1.1", "c2.1"],
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
    subdirectory = irl_path / "analysis" / inputs.experiment_subdirectory

    set_plotting_and_logging_defaults(
        subdirectory=subdirectory, experiment_name="Table", filename=Path(__file__).stem
    )

    analysis_obs = {}
    for experiment_name in inputs.experiment_names:
        analysis_obj = AnalysisObject(
            experiment_name,
            irl_path=irl_path,
            experiment_subdirectory=inputs.experiment_subdirectory,
        )
        analysis_obs[experiment_name] = analysis_obj.query_optimization_data(
            excluded_parameters=analysis_obj.analysis_details.excluded_parameters
        )

    model_params = list(
        set(analysis_obj.cost_details.constant_values)
        - set(analysis_obj.analysis_details.excluded_parameters)
    )

    for exp1, exp2 in list(combinations(inputs.experiment_names, 2)):
        logging.info("%s, %s", exp1, exp2)
        logging.info("--------------")
        for model_param in model_params:
            logging.info(model_param)
            stat_obj = pg.mwu(
                analysis_obs[exp1][model_param], analysis_obs[exp2][model_param]
            )
            logging.info(
                "M_{%s} = %.3f, M_{%s} = %.3f",
                exp1,
                np.mean((analysis_obs[exp1][model_param])),
                exp2,
                np.mean((analysis_obs[exp2][model_param])),
            )
            logging.info(get_mann_whitney_text(stat_obj))

    for exp in inputs.experiment_names:
        logging.info(exp)
        for model_param in model_params:
            logging.info(
                "%s & %.3f (%.3f) \\ ",  # noqa: W605
                analysis_obj.cost_details.latex_mapping[model_param],
                np.mean((analysis_obs[exp][model_param])),
                np.std((analysis_obs[exp][model_param])),
            )
