import itertools
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pingouin as pg
import seaborn as sns
from costometer.utils import get_correlation_text, standard_parse_args

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

    sim_latex_mapping = {
        f"sim_{k}": v for k, v in analysis_obj.cost_details.latex_mapping.items()
    }
    for subset in itertools.combinations(
        [f"sim_{param}" for param in analysis_obj.cost_details.latex_mapping.keys()],
        2,
    ):
        mean_over_sim_param = (
            optimization_data.groupby(list(subset), as_index=False).mean().reset_index()
        )
        mean_over_sim_param = mean_over_sim_param.rename(columns={"bic": "BIC"})

        pivoted_mean_ll = mean_over_sim_param.pivot(
            index=subset[0],
            columns=subset[1],
            values="BIC",
        )

        plt.figure(figsize=(12, 8), dpi=80)
        sns.heatmap(pivoted_mean_ll)
        plt.xlabel(f"${sim_latex_mapping[subset[1]]}$")
        plt.ylabel(f"${sim_latex_mapping[subset[0]]}$")
        plt.savefig(
            subdirectory.joinpath(
                f"figs/{inputs.experiment_name}_{'_'.join(sorted(subset))}xBIC.png"
            ),
            bbox_inches="tight",
        )

    for param in analysis_obj.cost_details.constant_values:
        logging.info("----------")
        logging.info("Correlation between {param} and BIC")
        logging.info("----------")
        correlation = pg.corr(optimization_data[param], optimization_data["bic"])
        logging.info(get_correlation_text(correlation))
