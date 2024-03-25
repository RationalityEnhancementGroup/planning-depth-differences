import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
from costometer.utils import (
    AnalysisObject,
    get_correlation_text,
    get_mann_whitney_text,
    get_wilcoxon_text,
    set_plotting_and_logging_defaults,
)
from sklearn import tree
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.model_selection import cross_validate
from statsmodels.regression.linear_model import OLSResults

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-e1",
        "--exp1",
        dest="experiment_name_fairy",
        default="ValidationExperimentBaseline",
        type=str,
    )
    parser.add_argument(
        "-e2",
        "--exp2",
        dest="experiment_name_test",
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
    subdirectory = irl_path.joinpath(f"analysis/{inputs.experiment_subdirectory}")
    set_plotting_and_logging_defaults(
        subdirectory=subdirectory,
        experiment_name="ValidationHDI",
        filename=Path(__file__).stem,
    )

    analysis_obj_test = AnalysisObject(
        inputs.experiment_name_test,
        irl_path=irl_path,
        experiment_subdirectory=inputs.experiment_subdirectory,
    )
    optimization_data_test = analysis_obj_test.query_optimization_data(
        excluded_parameters=analysis_obj_test.analysis_details.excluded_parameters
    )
    analysis_obj_fairy = AnalysisObject(
        inputs.experiment_name_fairy,
        irl_path=irl_path,
        experiment_subdirectory=inputs.experiment_subdirectory,
    )
    optimization_data_fairy = analysis_obj_fairy.query_optimization_data(
        excluded_parameters=analysis_obj_fairy.analysis_details.excluded_parameters
    )

    optimization_data_test["Block"] = "test"
    optimization_data_fairy["Block"] = "fairy"
    optimization_data = pd.concat([optimization_data_fairy, optimization_data_test])

    optimization_data = analysis_obj_test.join_optimization_df_and_processed(
        optimization_df=optimization_data,
        processed_df=analysis_obj_test.dfs["individual-variables"],
        variables_of_interest=["DEPTH", "COST", "FAIRY_GOD_CONDITION", "cond"],
    )

    model_params = list(
        set(analysis_obj_test.cost_details.constant_values)
        - set(analysis_obj_test.analysis_details.excluded_parameters)
    )

    model_params_given = {"depth_cost_weight": "DEPTH", "given_cost": "COST"}

    hdi_ranges = {}
    hdi_ranges["test"] = analysis_obj_test.load_hdi_ranges(
        excluded_parameter_str=analysis_obj_test.analysis_details.excluded_parameter_str
    )
    hdi_ranges["fairy"] = analysis_obj_fairy.load_hdi_ranges(
        excluded_parameter_str=analysis_obj_fairy.analysis_details.excluded_parameter_str  # noqa : E501
    )

    full_parameter_info = optimization_data.pivot(
        index="trace_pid",
        columns=["Block"],
        values=model_params + ["DEPTH", "COST", "FAIRY_GOD_CONDITION"],
    )

    min_and_max_data = {}
    for pid in analysis_obj_test.dfs["mouselab-mdp"]["pid"].unique():
        min_and_max_data[pid] = {
            **{
                f"{block}_{parameter}_max": hdi_ranges[block][pid][parameter][1]
                for block in hdi_ranges.keys()
                for parameter in hdi_ranges[block][pid].keys()
            },
            **{
                f"{block}_{parameter}_min": hdi_ranges[block][pid][parameter][0]
                for block in hdi_ranges.keys()
                for parameter in hdi_ranges[block][pid].keys()
            },
        }

    full_parameter_info = full_parameter_info.join(
        pd.DataFrame.from_dict(min_and_max_data, orient="index")
    )

    full_parameter_info.columns = [
        "_".join(cols) if isinstance(cols, tuple) else cols
        for cols in list(full_parameter_info)
    ]
    del full_parameter_info["DEPTH_fairy"]
    del full_parameter_info["COST_fairy"]
    del full_parameter_info["FAIRY_GOD_CONDITION_fairy"]
    full_parameter_info.rename(
        columns={
            "COST_test": "COST",
            "DEPTH_test": "DEPTH",
            "FAIRY_GOD_CONDITION_test": "FAIRY_GOD_CONDITION",
        },
        inplace=True,
    )

    logging.info("----------")
    for block in hdi_ranges.keys():
        for parameter in model_params:
            full_parameter_info[f"{block}_{parameter}_spread"] = (
                full_parameter_info[f"{block}_{parameter}_max"]
                - full_parameter_info[f"{block}_{parameter}_min"]
            )
        for parameter in model_params_given.keys():
            if len(parameter.split("_")) > 1:
                full_parameter_info[
                    f"{block}_{parameter}_in"
                ] = full_parameter_info.apply(
                    lambda row: (
                        row[model_params_given[parameter]]
                        <= row[f"{block}_{parameter}_max"]
                    )
                    and (
                        row[model_params_given[parameter]]
                        >= row[f"{block}_{parameter}_min"]
                    ),
                    axis=1,
                )
                logging.info(
                    "parameter: %s, block: %s in hdi (predictions) %.2f",
                    parameter,
                    block,
                    np.mean(full_parameter_info[f"{block}_{parameter}_in"]),
                )

    for param in model_params:
        res = pg.wilcoxon(
            full_parameter_info[f"fairy_{param}_spread"].astype(np.float64),
            full_parameter_info[f"test_{param}_spread"].astype(np.float64),
        )

        logging.info("----------")
        logging.info("Test vs fairy block spreads for parameter %s", param)
        logging.info("----------")
        logging.info(get_wilcoxon_text(res))
        logging.info(
            "test block ($M: %.2f, SD: %.2f$)\n"
            "baseline block ($M: %.2f, SD: %.2f$) ",
            full_parameter_info[f"test_{param}_spread"].mean(),
            full_parameter_info[f"test_{param}_spread"].std(),
            full_parameter_info[f"fairy_{param}_spread"].mean(),
            full_parameter_info[f"fairy_{param}_spread"].std(),
        )

    for block in hdi_ranges.keys():
        logging.info("----------")
        logging.info("Spread in %s block vs MAP error", block)
        logging.info("----------")
        for parameter in model_params_given.keys():
            full_parameter_info[f"diff_{parameter}"] = full_parameter_info.apply(
                lambda row: np.sqrt(
                    (row[f"{parameter}_{block}"] - row[model_params_given[parameter]])
                    ** 2
                ),
                axis=1,
            )

            correlation_object = pg.corr(
                full_parameter_info[f"{block}_{parameter}_spread"],
                full_parameter_info[f"diff_{parameter}"],
            )
            logging.info(
                "%s & %s",
                analysis_obj_test.cost_details.latex_mapping[parameter],
                get_correlation_text(correlation_object),
            )

        for parameter in model_params_given.keys():
            logging.info("%s, %s", parameter, block)
            if len(full_parameter_info[f"{block}_{parameter}_in"].unique()) == 1:
                logging.info(
                    "Unique: %d",
                    full_parameter_info[f"{block}_{parameter}_in"].unique(),
                )
            else:
                comparison = pg.mwu(
                    full_parameter_info[full_parameter_info[f"{block}_{parameter}_in"]][
                        f"diff_{parameter}"
                    ],
                    full_parameter_info[
                        ~full_parameter_info[f"{block}_{parameter}_in"]
                    ][f"diff_{parameter}"],
                )
                logging.info("%s", get_mann_whitney_text(comparison))

    full_parameter_info.rename(
        columns={f"{parameter}_test": parameter for parameter in model_params},
        inplace=True,
    )
    full_parameter_info.rename(
        columns={
            f"fairy_{parameter}": f"{parameter}_fairy" for parameter in model_params
        },
        inplace=True,
    )
    full_parameter_info["FAIRY_GOD_CONDITION"] = full_parameter_info[
        "FAIRY_GOD_CONDITION"
    ].astype(bool)
    for parameter in model_params:
        full_parameter_info[f"{parameter}_fairy"] = full_parameter_info[
            f"{parameter}_fairy"
        ].astype(np.float64)
        full_parameter_info[f"{parameter}"] = full_parameter_info[
            f"{parameter}"
        ].astype(np.float64)

    block = "test"
    for parameter, given_param in model_params_given.items():
        model = OLSResults.load(
            subdirectory.joinpath(f"data/regressions/{given_param}_model.pkl")
        )

        full_parameter_info[f"predictions_{parameter}"] = model.predict(
            full_parameter_info
        )
        full_parameter_info[f"diff_{parameter}"] = full_parameter_info.apply(
            lambda row: np.abs(row[f"predictions_{parameter}"] - row[given_param]),
            axis=1,
        )
        full_parameter_info[
            f"{block}_{parameter}_in_regression"
        ] = full_parameter_info.apply(
            lambda row: (
                row[f"predictions_{parameter}"] <= row[f"{block}_{parameter}_max"]
            )
            and (row[f"predictions_{parameter}"] >= row[f"{block}_{parameter}_min"]),
            axis=1,
        )
        logging.info(
            "%s, %s: in hdi (predictions, regression) %.2f",
            parameter,
            block,
            np.mean(full_parameter_info[f"{block}_{parameter}_in_regression"]),
        )

        if len(full_parameter_info[f"{block}_{parameter}_in_regression"].unique()) == 1:
            logging.info("----------")
            logging.info(
                "This should almost never happen, means "
                "true parameter is in spread 100% of time"
            )
            logging.info("----------")
            logging.info(
                "%d", full_parameter_info[f"{block}_{parameter}_in_regression"].unique()
            )
        else:
            comparison = pg.mwu(
                full_parameter_info[
                    full_parameter_info[f"{block}_{parameter}_in_regression"]
                    == True  # noqa : E712
                ][f"diff_{parameter}"],
                full_parameter_info[
                    full_parameter_info[f"{block}_{parameter}_in_regression"]
                    == False  # noqa : E712
                ][f"diff_{parameter}"],
            )
            logging.info("----------")
            logging.info(
                "Difference in spread when true parameter is contained vs not, "
                "block %s parameter %s",
                block,
                parameter,
            )
            logging.info("----------")
            logging.info("%s", get_mann_whitney_text(comparison))

    logging.info("----------")
    logging.info(
        "Correlation between parameter spread in test block and error in recovery"
    )
    logging.info("----------")
    for parameter in model_params_given.keys():
        correlation_object = pg.corr(
            full_parameter_info[f"{block}_{parameter}_spread"],
            full_parameter_info[f"diff_{parameter}"],
        )
        logging.info(
            "%s & %s",
            analysis_obj_test.cost_details.latex_mapping[parameter],
            get_correlation_text(correlation_object, table=True),
        )

    logging.info("----------")
    logging.info(
        "Correlation between inferred temperature and parameter spread in %s block",
        block,
    )
    logging.info("----------")
    for parameter in model_params:
        correlation_object = pg.corr(
            full_parameter_info["temp"],
            full_parameter_info[f"{block}_{parameter}_spread"],
        )
        logging.info(
            "%s & %s",
            analysis_obj_test.cost_details.latex_mapping[parameter],
            get_correlation_text(correlation_object, table=True),
        )

    logging.info("----------")
    logging.info("Correlation between inferred temperature and error in recovery")
    logging.info("----------")
    for parameter in model_params_given.keys():
        correlation_object = pg.corr(
            full_parameter_info["temp"],
            full_parameter_info[f"diff_{parameter}"],
        )
        logging.info(
            "%s & %s",
            analysis_obj_test.cost_details.latex_mapping[parameter],
            get_correlation_text(correlation_object, table=True),
        )

    logging.info("----------")
    logging.info(
        "Correlation between regression estimate and parameter spread in test block"
    )
    logging.info("----------")
    for parameter in model_params_given.keys():
        correlation_object = pg.corr(
            full_parameter_info[f"predictions_{parameter}"],
            full_parameter_info[f"{block}_{parameter}_spread"],
        )
        logging.info(
            "%s & %s",
            analysis_obj_test.cost_details.latex_mapping[parameter],
            get_correlation_text(correlation_object, table=True),
        )

    logging.info("----------")
    logging.info(
        "Correlation between given parameter and parameter spread in test block"
    )
    logging.info("----------")
    for parameter in model_params_given.keys():
        correlation_object = pg.corr(
            full_parameter_info[given_param].astype(np.float64),
            full_parameter_info[f"{block}_{parameter}_spread"],
        )
        logging.info(
            "%s & %s",
            analysis_obj_test.cost_details.latex_mapping[parameter],
            get_correlation_text(correlation_object, table=True),
        )

    logging.info("----------")
    logging.info("Number of participants: %d", len(full_parameter_info))
    logging.info("----------")

    logging.info("----------")
    logging.info("Spread in test block vs MAP")
    logging.info("----------")
    for parameter in model_params:
        correlation_object = pg.corr(
            full_parameter_info[f"test_{parameter}_spread"],
            full_parameter_info[f"{parameter}"].astype(np.float64),
        )
        logging.info(
            "%s & %s",
            analysis_obj_test.cost_details.latex_mapping[parameter],
            get_correlation_text(correlation_object, table=True),
        )

    logging.info("----------")
    logging.info("Correlation between linear regression output and HDI min")
    logging.info("----------")
    for parameter in model_params_given.keys():
        correlation_object = pg.corr(
            full_parameter_info[f"predictions_{parameter}"],
            full_parameter_info[f"{block}_{parameter}_min"],
        )
        logging.info(
            "%s & %s",
            analysis_obj_test.cost_details.latex_mapping[parameter],
            get_correlation_text(correlation_object, table=True),
        )

    logging.info("----------")
    logging.info("Correlation between linear regression output and HDI max")
    logging.info("----------")
    for parameter in model_params_given.keys():
        correlation_object = pg.corr(
            full_parameter_info[f"predictions_{parameter}"],
            full_parameter_info[f"{block}_{parameter}_max"],
        )
        logging.info(
            "%s & %s",
            analysis_obj_test.cost_details.latex_mapping[parameter],
            get_correlation_text(correlation_object, table=True),
        )

    for parameter in model_params_given.keys():
        threshold = np.median(full_parameter_info[f"{model_params_given[parameter]}"])
        logging.info("----------")
        logging.info("%s, %s", parameter, threshold)
        logging.info("----------")
        y_true = (
            full_parameter_info[f"{model_params_given[parameter]}"] >= threshold
        ).values
        y_pred = (full_parameter_info[f"predictions_{parameter}"]).values
        clf = tree.DecisionTreeClassifier(max_depth=1)
        # clf = clf.fit(y_pred.reshape( -1,1), y_true)
        res = cross_validate(
            clf,
            X=y_pred.reshape(-1, 1),
            y=y_true,
            scoring=make_scorer(balanced_accuracy_score),
            return_estimator=True,
        )
        logging.info("----------")
        logging.info("Avg cross val score: %.2f", res["test_score"].mean())
        logging.info("----------")
        logging.info(
            "%s", ",".join([tree.export_text(est) for est in res["estimator"]])
        )
