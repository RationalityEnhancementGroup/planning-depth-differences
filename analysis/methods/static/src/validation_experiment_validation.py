from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
import yaml
from costometer.utils import (
    AnalysisObject,
    get_anova_text,
    get_correlation_text,
    get_mann_whitney_text,
    get_trajectories_from_participant_data,
    traces_to_df,
)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp",
        dest="experiment_name",
        default="ValidationCostModel",
        type=str,
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
        default="linear_depth",
        type=str,
    )
    inputs = parser.parse_args()

    data_path = Path(__file__).resolve().parents[1]
    irl_path = Path(__file__).resolve().parents[4]

    analysis_obj = AnalysisObject(
        inputs.experiment_name,
        irl_path=irl_path,
        experiment_subdirectory=inputs.experiment_subdirectory,
    )

    optimization_data = analysis_obj.add_individual_variables(
        analysis_obj.query_optimization_data(),
        variables_of_interest=["DEPTH", "COST", "FAIRY_GOD_CONDITION", "cond"],
    )
    optimization_data = optimization_data[
        optimization_data["Model Name"] == "Effort Cost and Planning Depth"
    ]

    # read in experiment setting variables
    yaml_path = irl_path.joinpath(
        f"data/inputs/yamls/experiment_settings/{analysis_obj.experiment_setting}.yaml"
    )
    with open(yaml_path, "r") as stream:
        experiment_details = yaml.safe_load(stream)

    participant_trajectory = traces_to_df(
        get_trajectories_from_participant_data(
            mouselab_mdp_dataframe=analysis_obj.mouselab_trials,
            experiment_setting=analysis_obj.experiment_setting,
        )
    )

    for classification, nodes in experiment_details["node_classification"].items():
        participant_trajectory[classification] = participant_trajectory[
            "actions"
        ].apply(lambda action: action in nodes)

    analysis_obj.mouselab_trials = pd.merge(
        analysis_obj.mouselab_trials,
        participant_trajectory.groupby(["pid", "i_episode"])
        .sum()[experiment_details["node_classification"]]
        .reset_index(),
        left_on=["pid", "trial_index"],
        right_on=["pid", "i_episode"],
    )

    node_classification_per_block = analysis_obj.mouselab_trials.groupby(
        ["block", "pid"], as_index=False
    ).mean()[
        ["block", "pid", "FAIRY_GOD_CONDITION"]
        + list(experiment_details["node_classification"].keys())
    ]

    results_df = node_classification_per_block.merge(
        optimization_data[
            ["pid", "DEPTH", "COST", "temp", "depth_cost_weight", "static_cost_weight"]
        ]
    )

    results_df["pctg_late"] = results_df.apply(
        lambda row: row["late"] / (row["clicks"] + np.finfo(float).eps), axis=1
    )

    for metric in ["pctg_late", "clicks"]:
        print("----------")
        print(f"Difference in behavior between test and fairy blocks: {metric}")
        print("----------")
        comparison = pg.wilcoxon(
            results_df[results_df["block"] == "test"].sort_values(["pid"])[metric],
            results_df[results_df["block"] == "fairy"].sort_values(["pid"])[metric],
        )

        print(comparison)

    print("==========")
    for analysis_pair in [("pctg_late", "DEPTH"), ("clicks", "COST")]:
        dv, between = analysis_pair
        print("----------")
        print(f"ANOVA results for dv: {dv}, between: {between}, within: block")
        print("----------")
        anova_object = pg.mixed_anova(
            data=results_df[results_df["block"].isin(["test", "fairy"])],
            dv=dv,
            within="block",
            between=between,
            subject="pid",
        )
        print(get_anova_text(anova_object))

        print("----------")
        print(f"ANOVA results for dv: {dv}, between: block order, within: block")
        print("----------")
        anova_object = pg.mixed_anova(
            data=results_df[results_df["block"].isin(["test", "fairy"])],
            dv=dv,
            within="block",
            between="FAIRY_GOD_CONDITION",
            subject="pid",
        )
        print(get_anova_text(anova_object))

        print("==========")
        for block in results_df["block"].unique():
            print("----------")
            print(f"Correlation between {dv} and {between} for block: {block}")
            print("----------")
            correlation_obj = pg.corr(
                results_df[results_df["block"] == block][dv],
                results_df[results_df["block"] == block][between],
                method="spearman",
            )
            print(get_correlation_text(correlation_obj))

    for block in results_df["block"].unique():
        curr_result_df = results_df[results_df["block"] == block]

        print(f"Difference in block order for clicks in {block} block")
        comparison = pg.mwu(
            curr_result_df[curr_result_df["FAIRY_GOD_CONDITION"] == True][  # noqa: E712
                "clicks"
            ],
            curr_result_df[curr_result_df["FAIRY_GOD_CONDITION"] == False]["clicks"],
        )
        print(get_mann_whitney_text(comparison))
