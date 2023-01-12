from argparse import ArgumentParser
from pathlib import Path

import dill as pickle
import numpy as np
import pandas as pd
import pingouin as pg
from costometer.utils import (
    AnalysisObject,
    get_correlation_text,
    get_mann_whitney_text,
    get_wilcoxon_text,
)
from sklearn import tree
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.model_selection import cross_validate
from statsmodels.regression.linear_model import OLSResults

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
        dest="experiment_subdirectory",
        metavar="experiment_subdirectory",
    )
    parser.add_argument(
        "-c",
        "--cost-function",
        dest="cost_function",
        default="dist_depth_eff_forw",
        type=str,
    )
    inputs = parser.parse_args()

    irl_path = Path(__file__).resolve().parents[4]
    data_path = irl_path.joinpath(f"analysis/{inputs.experiment_subdirectory}")

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
        optimization_data["Model Name"] == "'Distance, Effort, Depth and Forward Search Bonus'"
    ]

    model_params = analysis_obj.cost_details[inputs.cost_function][
        "cost_parameter_args"
    ] + ["temp"]

    model_params_given = {"depth_cost_weight": "DEPTH", "given_cost": "COST"}

    hdi_file = data_path.joinpath(
        f"data/{inputs.experiment_name}/"
        f"{inputs.experiment_name}_{inputs.cost_function}_hdi.pickle"
    )

    with open(
        hdi_file,
        "rb",
    ) as f:
        hdi_ranges = pickle.load(f)

    full_parameter_info = optimization_data.pivot(
        index="trace_pid",
        columns=["Block"],
        values=model_params + ["DEPTH", "COST", "FAIRY_GOD_CONDITION"],
    )

    for parameter in model_params:
        max_data = pd.DataFrame(
            {
                f"{block}_{parameter}_max": {
                    key: val[1] for key, val in hdi_ranges[block][parameter].items()
                }
                for block in hdi_ranges.keys()
            }
        )
        min_data = pd.DataFrame(
            {
                f"{block}_{parameter}_min": {
                    key: val[0] for key, val in hdi_ranges[block][parameter].items()
                }
                for block in hdi_ranges.keys()
            }
        )

        full_parameter_info = full_parameter_info.join(max_data)
        full_parameter_info = full_parameter_info.join(min_data)

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

    for block in hdi_ranges.keys():
        for parameter in model_params:
            full_parameter_info[f"{block}_{parameter}_spread"] = (
                full_parameter_info[f"{block}_{parameter}_max"]
                - full_parameter_info[f"{block}_{parameter}_min"]
            )
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
                print(
                    parameter,
                    block,
                    f"in hdi (predictions) "
                    f"{np.mean(full_parameter_info[f'{block}_{parameter}_in']):.2f}",
                )

    for param in model_params:
        res = pg.wilcoxon(
            full_parameter_info[f"fairy_{param}_spread"].astype(np.float),
            full_parameter_info[f"test_{param}_spread"].astype(np.float),
        )

        print("----------")
        print(f"Test vs fairy block spreads for parameter {param}")
        print("----------")
        print(get_wilcoxon_text(res))
        print(
            f"test block ($M: {full_parameter_info[f'test_{param}_spread'].mean():.2f},"
            f" SD: {full_parameter_info[f'test_{param}_spread'].std():.2f}$)\n"
            f"baseline block ($M: {full_parameter_info[f'fairy_{param}_spread'].mean():.2f},"  # noqa: E501
            f" SD: {full_parameter_info[f'fairy_{param}_spread'].std():.2f}$) "
        )

    for block in hdi_ranges.keys():
        for parameter in analysis_obj.cost_details[inputs.cost_function][
            "cost_parameter_args"
        ]:
            full_parameter_info[f"diff_{parameter}"] = full_parameter_info.apply(
                lambda row: np.sqrt(
                    (row[f"{parameter}_{block}"] - row[model_params_given[parameter]])
                    ** 2
                ),
                axis=1,
            )
            print(parameter, block)
            if len(full_parameter_info[f"{block}_{parameter}_in"].unique()) == 1:
                print(full_parameter_info[f"{block}_{parameter}_in"].unique())
            else:

                comparison = pg.mwu(
                    full_parameter_info[
                        full_parameter_info[f"{block}_{parameter}_in"]
                        == True  # noqa: E712, E501
                    ][
                        f"diff_{parameter}"
                    ],  # noqa
                    full_parameter_info[
                        full_parameter_info[f"{block}_{parameter}_in"] == False
                    ][f"diff_{parameter}"],
                )
                print(get_mann_whitney_text(comparison))

            print(f"Spread in {block} block vs MAP error {parameter}")
            correlation_object = pg.corr(
                full_parameter_info[f"{block}_{parameter}_spread"],
                full_parameter_info[f"diff_{parameter}"],
            )
            print(get_correlation_text(correlation_object))

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
    for parameter in analysis_obj.cost_details[inputs.cost_function][
        "cost_parameter_args"
    ]:
        given_param = model_params_given[parameter]
        model = OLSResults.load(
            data_path.joinpath(f"data/regressions/{given_param}_model.pkl")
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
        print(
            parameter,
            block,
            f"in hdi (predictions, regression) "
            f"{np.mean(full_parameter_info[f'{block}_{parameter}_in_regression']):.2f}",
        )

        if len(full_parameter_info[f"{block}_{parameter}_in_regression"].unique()) == 1:
            print("----------")
            print(
                "This should almost never happen, means "
                "true parameter is in spread 100% of time"
            )
            print("----------")
            print(full_parameter_info[f"{block}_{parameter}_in_regression"].unique())
        else:
            comparison = pg.mwu(
                full_parameter_info[
                    full_parameter_info[f"{block}_{parameter}_in_regression"]
                    == True  # noqa
                ][f"diff_{parameter}"],
                full_parameter_info[
                    full_parameter_info[f"{block}_{parameter}_in_regression"] == False
                ][f"diff_{parameter}"],
            )
            print("----------")
            print(
                f"Difference in spread when true parameter is "
                f"contained vs not, block {block} parameter {parameter}"
            )
            print("----------")
            print(get_mann_whitney_text(comparison))

    for parameter in analysis_obj.cost_details[inputs.cost_function][
        "cost_parameter_args"
    ]:
        print("----------")
        print(
            f"Correlation between parameter spread in "
            f"test block and error in recovery, {parameter}"
        )
        print("----------")
        correlation_object = pg.corr(
            full_parameter_info[f"{block}_{parameter}_spread"],
            full_parameter_info[f"diff_{parameter}"],
        )
        print(get_correlation_text(correlation_object))

        print("----------")
        print(
            f"Correlation between inferred temperature and "
            f"parameter spread in {block} block, {parameter}"
        )
        print("----------")
        correlation_object = pg.corr(
            full_parameter_info["temp"],
            full_parameter_info[f"{block}_{parameter}_spread"],
        )
        print(get_correlation_text(correlation_object))

        print("----------")
        print(
            f"Correlation between inferred temperature "
            f"and error in recovery, {parameter}"
        )
        print("----------")
        correlation_object = pg.corr(
            full_parameter_info["temp"],
            full_parameter_info[f"diff_{parameter}"],
        )
        print(get_correlation_text(correlation_object))

        print("----------")
        print(
            f"Correlation between regression estimate {given_param} "
            f"and parameter spread in test block, {parameter}"
        )
        print("----------")
        correlation_object = pg.corr(
            full_parameter_info[f"predictions_{parameter}"],
            full_parameter_info[f"{block}_{parameter}_spread"],
        )
        print(get_correlation_text(correlation_object))

        print("----------")
        print(
            f"Correlation between given parameter {given_param} "
            f"and parameter spread in test block, {parameter}"
        )
        print("----------")
        correlation_object = pg.corr(
            full_parameter_info[given_param].astype(np.float64),
            full_parameter_info[f"{block}_{parameter}_spread"],
        )

        print(get_correlation_text(correlation_object))

    print("----------")
    print(f"Number of participants: {len(full_parameter_info)}")

    for parameter in model_params:
        print(f"Spread in test block vs MAP for {parameter}")
        correlation_object = pg.corr(
            full_parameter_info[f"test_{parameter}_spread"],
            full_parameter_info[f"{parameter}"].astype(np.float64),
        )
        print(get_correlation_text(correlation_object))

    for parameter in analysis_obj.cost_details[inputs.cost_function][
        "cost_parameter_args"
    ]:
        correlation_object = pg.corr(
            full_parameter_info[f"predictions_{parameter}"],
            full_parameter_info[f"{block}_{parameter}_min"],
        )
        print(f"Correlation between linear regression output and HDI min {parameter}")
        print(get_correlation_text(correlation_object))

        correlation_object = pg.corr(
            full_parameter_info[f"predictions_{parameter}"],
            full_parameter_info[f"{block}_{parameter}_max"],
        )
        print(f"Correlation between linear regression output and HDI max {parameter}")
        print(get_correlation_text(correlation_object))

    for parameter in analysis_obj.cost_details[inputs.cost_function][
        "cost_parameter_args"
    ]:

        threshold = np.median(full_parameter_info[f"{model_params_given[parameter]}"])
        print("----------")
        print(f"{parameter}, {threshold}")
        print("----------")
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
        # print(res)
        print(f"Avg cross val score: {res['test_score'].mean():.2f}")
        print("----------")
        print(",".join([tree.export_text(est) for est in res["estimator"]]))
