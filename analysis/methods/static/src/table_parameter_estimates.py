import logging
from argparse import ArgumentParser
from itertools import combinations
from pathlib import Path

import numpy as np
import pingouin as pg
from costometer.utils import AnalysisObject, get_mann_whitney_text

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
    subdirectory = irl_path.joinpath(f"analysis/{inputs.experiment_subdirectory}")

    analysis_obs = {}
    for experiment_name in inputs.experiment_names:
        analysis_obj = AnalysisObject(
            experiment_name,
            irl_path=irl_path,
            experiment_subdirectory=inputs.experiment_subdirectory,
        )
        analysis_obs[experiment_name] = analysis_obj.query_optimization_data(
            excluded_parameters=analysis_obj.excluded_parameters
        )

    model_params = list(
        set(analysis_obj.cost_details["constant_values"])
        - set(analysis_obj.excluded_parameters.split(","))
    )

    for exp1, exp2 in list(combinations(inputs.experiment_names, 2)):
        logging.info(f"{exp1}, {exp2}")
        logging.info("--------------")
        for model_param in model_params:
            logging.info(model_param)
            stat_obj = pg.mwu(
                analysis_obs[exp1][model_param], analysis_obs[exp2][model_param]
            )
            logging.info(
                f"M_{{{exp1}}} = "
                f"{np.mean((analysis_obs[exp1][model_param])):.3f}"
                f", M_{{{exp2}}} = "
                f"{np.mean((analysis_obs[exp2][model_param])):.3f}"
            )
            logging.info(get_mann_whitney_text(stat_obj))

    for exp in inputs.experiment_names:
        logging.info(exp)
        for model_param in model_params:
            logging.info(
                f"{analysis_obj.cost_details['latex_mapping'][model_param]} & {np.mean((analysis_obs[exp][model_param])):.3f} ({np.std((analysis_obs[exp][model_param])):.3f}) \\\ "  # noqa: W605, E501
            )
