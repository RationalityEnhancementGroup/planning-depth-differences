from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import pingouin as pg
from costometer.utils import AnalysisObject, get_correlation_text, set_font_sizes

set_font_sizes()

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

    for excluded_parameters in analysis_obj.trial_by_trial_models:
        print(excluded_parameters)
        print("================================")
        curr_optimization_data = analysis_obj.query_optimization_data(
            excluded_parameters=excluded_parameters
        )

        curr_merged_df = analysis_obj.join_optimization_df_and_processed(
            optimization_df=curr_optimization_data,
            processed_df=analysis_obj.dfs["mouselab-mdp"],
            variables_of_interest=list(analysis_obj.cost_details["constant_values"]),
        )

        sum_clicks = (
            curr_merged_df.groupby(list(analysis_obj.cost_details["constant_values"]))
            .mean()
            .reset_index()
        )

        sum_over_pids = (
            curr_merged_df.groupby(
                [
                    "pid",
                ]
                + list(analysis_obj.cost_details["constant_values"])
            )
            .mean()
            .reset_index()
        )

        subdirectory.joinpath("processed/human").mkdir(parents=True, exist_ok=True)
        sum_over_pids.to_csv(
            subdirectory.joinpath(f"processed/human/{inputs.experiment_name}_bias.csv")
        )

        simulated_df = pd.read_csv(
            irl_path.joinpath(
                f"cluster/data/trajectories/{analysis_obj.experiment_setting}"
                f"/SoftmaxPolicy/{analysis_obj.simulated_param_run}"
                f"{'_' + excluded_parameters if excluded_parameters != '' else excluded_parameters}"
                f"/simulated_agents_back_dist_depth_eff_forw.csv"
            )
        )

        # add node classification columns
        for classification, nodes in analysis_obj.experiment_details[
            "node_classification"
        ].items():
            simulated_df[f"num_{classification}"] = simulated_df["actions"].apply(
                lambda action: action in nodes
            )

        mean_over_cost = (
            simulated_df.groupby(
                ["pid", "i_episode"]
                + [
                    f"sim_{param}"
                    for param in analysis_obj.cost_details["constant_values"]
                ]
            )
            .sum()
            .reset_index()
            .groupby(
                [
                    f"sim_{param}"
                    for param in analysis_obj.cost_details["constant_values"]
                ]
            )
            .mean()
            .reset_index()
        )
        sum_over_pids = sum_over_pids.merge(
            mean_over_cost,
            left_on=list(analysis_obj.cost_details["constant_values"]),
            right_on=[
                f"sim_{param}" for param in analysis_obj.cost_details["constant_values"]
            ],
            suffixes=("", "_optimal"),
        )

        for classification, nodes in analysis_obj.experiment_details[
            "node_classification"
        ].items():
            print(
                f"Correlation of metric '{classification}' between "
                f"simulated and real data, per participant"
            )
            correlation_obj = pg.corr(
                sum_over_pids[f"num_{classification}"],
                sum_over_pids[f"num_{classification}_optimal"],
            )
            print(get_correlation_text(correlation_obj))

        sum_over_params = (
            sum_over_pids.groupby(list(analysis_obj.cost_details["constant_values"]))
            .mean()
            .reset_index()
        )
        for classification, nodes in analysis_obj.experiment_details[
            "node_classification"
        ].items():
            print(
                f"Correlation of metric '{classification}' between "
                f"simulated and real data, per cost setting"
            )
            correlation_obj = pg.corr(
                sum_over_params[f"num_{classification}"],
                sum_over_params[f"num_{classification}_optimal"],
            )
            print(get_correlation_text(correlation_obj))
