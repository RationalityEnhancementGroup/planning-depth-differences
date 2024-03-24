import itertools
import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
from costometer.utils import (
    AnalysisObject,
    get_correlation_text,
    get_friedman_test_text,
    get_pval_string,
)

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
    subdirectory = irl_path.joinpath(f"analysis/{inputs.experiment_subdirectory}")

    analysis_obj = AnalysisObject(
        inputs.experiment_name,
        irl_path=irl_path,
        experiment_subdirectory=inputs.experiment_subdirectory,
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
                f"Correlation of metric '{classification}' between "
                f"simulated and real data, per participant"
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

    for (
        classification,
        nodes,
    ) in analysis_obj.experiment_details.node_classification.items():
        logging.info(f"Full Friedman {classification}")
        friedman_object = pg.friedman(
            dv=f"difference_{classification}",
            within="excluded",  # TODO: why
            subject="pid",
            data=all_cost_model_df,
        )
        logging.info(get_friedman_test_text(friedman_object))
        logging.info(friedman_object)

    for (
        classification,
        nodes,
    ) in analysis_obj.experiment_details.node_classification.items():
        logging.info(f"Pair-wise table for {classification}")
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
            logging.info(f"Descriptive {model}, {classification} difference")
            descriptive_stats = all_cost_model_df[
                all_cost_model_df["excluded"] == model
            ][f"difference_{classification}"].describe()
            logging.info(
                f"M={descriptive_stats['mean']:.3f}, SD={descriptive_stats['std']:.3f}"
            )
