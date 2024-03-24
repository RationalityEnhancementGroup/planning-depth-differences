import logging
import sys
from pathlib import Path

import numpy as np
import pingouin as pg
from costometer.utils import (
    get_anova_text,
    get_correlation_text,
    get_mann_whitney_text,
    standard_parse_args,
)

if __name__ == "__main__":
    irl_path = Path(__file__).resolve().parents[4]
    analysis_obj, inputs, subdirectory = standard_parse_args(
        description=sys.modules[__name__].__doc__,
        irl_path=irl_path,
        filename=Path(__file__).stem,
        default_experiment="ValidationExperiment",
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
            for node_classification in analysis_obj.experiment_details.node_classification.keys()  # noqa : E501
        ]
    ]

    for metric in ["pctg_late", "num_nodes"]:
        logging.info("----------")
        logging.info("Difference in behavior between test and fairy blocks: {metric}")
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
        logging.info("ANOVA results for dv: {dv}, between: {between}, within: block")
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
        logging.info("ANOVA results for dv: {dv}, between: block order, within: block")
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
            logging.info("Correlation between {dv} and {between} for block: {block}")
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

        logging.info("Difference in block order for clicks in {block} block")
        comparison = pg.mwu(
            curr_result_df[curr_result_df["FAIRY_GOD_CONDITION"] == 1]["num_clicks"],
            curr_result_df[curr_result_df["FAIRY_GOD_CONDITION"] == 0]["num_clicks"],
        )
        logging.info(get_mann_whitney_text(comparison))
