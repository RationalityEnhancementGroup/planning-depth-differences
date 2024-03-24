import itertools
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
from costometer.utils import (
    get_correlation_text,
    get_friedman_test_text,
    get_pval_string,
    standard_parse_args,
)

if __name__ == "__main__":
    irl_path = Path(__file__).resolve().parents[4]
    analysis_obj, inputs, subdirectory = standard_parse_args(
        description=sys.modules[__name__].__doc__,
        irl_path=irl_path,
        filename=Path(__file__).stem,
    )

    all_cost_model_df = []
    for (
        excluded_parameter_string,
        excluded_parameters,
    ) in analysis_obj.analysis_details.trial_by_trial_models_mapping.items():
        logging.info(excluded_parameters)
        logging.info("================================")

        curr_optimization_data = analysis_obj.query_optimization_data(
            excluded_parameters=excluded_parameters
        )

        curr_merged_df = analysis_obj.join_optimization_df_and_processed(
            optimization_df=curr_optimization_data,
            processed_df=analysis_obj.dfs["mouselab-mdp"],
            variables_of_interest=list(analysis_obj.cost_details.constant_values),
        )

        sum_over_pids = (
            curr_merged_df.groupby(
                [
                    "pid",
                ]
                + list(analysis_obj.cost_details.constant_values)
            )
            .mean()
            .reset_index()
        )

        simulated_df = pd.read_csv(
            irl_path.joinpath(
                f"cluster/data/trajectories/{analysis_obj.experiment_setting}"
                f"/SoftmaxPolicy/{analysis_obj.analysis_details.simulated_param_run}"
                f"{'_' + excluded_parameter_string if excluded_parameter_string != '' else excluded_parameter_string}"  # noqa : E501
                f"/simulated_agents_back_dist_depth_eff_forw.csv"
            )
        )

        # add node classification columns
        for (
            classification,
            nodes,
        ) in analysis_obj.experiment_details.node_classification.items():
            simulated_df[f"num_{classification}"] = simulated_df["actions"].apply(
                lambda action: action in nodes
            )

        mean_over_cost = (
            simulated_df.groupby(
                ["pid", "i_episode"]
                + [
                    f"sim_{param}"
                    for param in analysis_obj.cost_details.constant_values
                ]
            )
            .sum()
            .reset_index()
            .groupby(
                [f"sim_{param}" for param in analysis_obj.cost_details.constant_values]
            )
            .mean()
            .reset_index()
        )
        sum_over_pids = sum_over_pids.merge(
            mean_over_cost,
            left_on=list(analysis_obj.cost_details.constant_values),
            right_on=[
                f"sim_{param}" for param in analysis_obj.cost_details.constant_values
            ],
            suffixes=("", "_optimal"),
        )
        sum_over_pids["excluded"] = excluded_parameter_string

        for (
            classification
        ) in analysis_obj.experiment_details.node_classification.keys():
            logging.info(
                "Correlation of metric '%s' between simulated and real data,"
                " per participant",
                {classification},
            )
            correlation_obj = pg.corr(
                sum_over_pids[f"num_{classification}"],
                sum_over_pids[f"num_{classification}_optimal"],
            )
            logging.info(get_correlation_text(correlation_obj))

            # A little hacky, for the supplementary analyses
            curr_df = sum_over_pids.copy(deep=True)
            curr_df[f"difference_{classification}"] = curr_df.apply(
                lambda row: np.abs(
                    row[f"num_{classification}"] - row[f"num_{classification}_optimal"]
                ),
                axis=1,
            )
            # curr_df["classification"] = classification
            all_cost_model_df.append(
                curr_df[[f"difference_{classification}", "pid", "excluded"]]
            )

    all_cost_model_df = pd.concat(all_cost_model_df)

    for classification in analysis_obj.experiment_details.node_classification.keys():
        logging.info("Full Friedman %s", classification)
        friedman_object = pg.friedman(
            dv=f"difference_{classification}",
            within="excluded",  # TODO: why
            subject="pid",
            data=all_cost_model_df,
        )
        logging.info(get_friedman_test_text(friedman_object))
        logging.info(friedman_object)

    for classification in analysis_obj.experiment_details.node_classification.keys():
        logging.info("Pair-wise table for %s", classification)
        for model_pair in itertools.combinations(
            analysis_obj.analysis_details.trial_by_trial_models, 2
        ):
            friedman_object = pg.friedman(
                dv=f"difference_{classification}",
                within="excluded",
                subject="pid",
                data=all_cost_model_df[all_cost_model_df["excluded"].isin(model_pair)],
            )

            logging.info(
                f"{analysis_obj.model_name_mapping[() if model_pair[0] == '' else tuple(model_pair[0].split(','))]} & "  # noqa : E501
                f"{analysis_obj.model_name_mapping[() if model_pair[1] == '' else tuple(model_pair[1].split(','))]} & "  # noqa : E501
                f"{friedman_object.Q[0]:.3f}"
                f"{get_pval_string(friedman_object['p-unc'][0])} \\"
            )

        for model_pair in itertools.combinations(
            analysis_obj.analysis_details.trial_by_trial_models, 2
        ):
            friedman_object = pg.friedman(
                dv=f"difference_{classification}",
                within="excluded",
                subject="pid",
                data=all_cost_model_df[all_cost_model_df["excluded"].isin(model_pair)],
            )

            logging.info(get_friedman_test_text(friedman_object))

    for classification in analysis_obj.experiment_details.node_classification.keys():
        for model in analysis_obj.analysis_details.trial_by_trial_models:
            logging.info("Descriptive %s, %s difference", model, classification)
            descriptive_stats = all_cost_model_df[
                all_cost_model_df["excluded"] == model
            ][f"difference_{classification}"].describe()
            logging.info(
                "M=%.3f, SD=%.3f", descriptive_stats["mean"], descriptive_stats["std"]
            )
