import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pingouin as pg
from costometer.utils import (
    AnalysisObject,
    get_anova_text,
    get_correlation_text,
    get_mann_whitney_text,
)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp",
        dest="experiment_name",
        default="ValidationExperiment",
        type=str,
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

    mouselab_data = analysis_obj.dfs["mouselab-mdp"]
    mouselab_data["pctg_late"] = mouselab_data.apply(
        lambda row: row["num_late"] / (row["num_nodes"] + np.finfo(float).eps), axis=1
    )

    node_classification_per_block = mouselab_data.groupby(
        ["block", "pid"], as_index=False
    ).mean()[
        [
            "block",
            "pid",
            "FAIRY_GOD_CONDITION",
            "DEPTH",
            "COST",
            "pctg_late",
            "num_nodes",
        ]
        + [
            f"num_{node_classification}"
            for node_classification in analysis_obj.experiment_details.node_classification.keys()
        ]
    ]

    for metric in ["pctg_late", "num_nodes"]:
        logging.info("----------")
        logging.info(f"Difference in behavior between test and fairy blocks: {metric}")
        logging.info("----------")
        comparison = pg.wilcoxon(
            node_classification_per_block[
                node_classification_per_block["block"] == "test"
            ].sort_values(["pid"])[metric],
            node_classification_per_block[
                node_classification_per_block["block"] == "fairy"
            ].sort_values(["pid"])[metric],
        )

        logging.info(comparison)

    logging.info("==========")
    for analysis_pair in [("pctg_late", "DEPTH"), ("num_nodes", "COST")]:
        dv, between = analysis_pair
        logging.info("----------")
        logging.info(f"ANOVA results for dv: {dv}, between: {between}, within: block")
        logging.info("----------")
        anova_object = pg.mixed_anova(
            data=node_classification_per_block[
                node_classification_per_block["block"].isin(["test", "fairy"])
            ],
            dv=dv,
            within="block",
            between=between,
            subject="pid",
        )
        logging.info(get_anova_text(anova_object))

        logging.info("----------")
        logging.info(f"ANOVA results for dv: {dv}, between: block order, within: block")
        logging.info("----------")
        anova_object = pg.mixed_anova(
            data=node_classification_per_block[
                node_classification_per_block["block"].isin(["test", "fairy"])
            ],
            dv=dv,
            within="block",
            between="FAIRY_GOD_CONDITION",
            subject="pid",
        )
        logging.info(get_anova_text(anova_object))

        logging.info("==========")
        for block in node_classification_per_block["block"].unique():
            logging.info("----------")
            logging.info(f"Correlation between {dv} and {between} for block: {block}")
            logging.info("----------")
            correlation_obj = pg.corr(
                node_classification_per_block[
                    node_classification_per_block["block"] == block
                ][dv],
                node_classification_per_block[
                    node_classification_per_block["block"] == block
                ][between],
                method="spearman",
            )
            logging.info(get_correlation_text(correlation_obj))

    for block in node_classification_per_block["block"].unique():
        curr_result_df = node_classification_per_block[
            node_classification_per_block["block"] == block
        ]

        logging.info(f"Difference in block order for clicks in {block} block")
        comparison = pg.mwu(
            curr_result_df[curr_result_df["FAIRY_GOD_CONDITION"] == 1]["num_clicks"],
            curr_result_df[curr_result_df["FAIRY_GOD_CONDITION"] == 0]["num_clicks"],
        )
        logging.info(get_mann_whitney_text(comparison))
