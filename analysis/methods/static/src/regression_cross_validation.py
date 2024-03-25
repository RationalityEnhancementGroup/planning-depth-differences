import logging
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pingouin as pg
import seaborn as sns
import statsmodels.formula.api as smf
from costometer.utils import (
    AnalysisObject,
    get_mann_whitney_text,
    get_regression_text,
    get_ttest_text,
    get_wilcoxon_text,
    set_plotting_and_logging_defaults,
)
from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp",
        dest="main_experiment_name",
        default="MainExperiment",
        type=str,
    )
    parser.add_argument(
        "-e1",
        "--exp1",
        dest="experiment_name_baseline",
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
        experiment_name="ValidationRegression",
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

    analysis_obj_baseline = AnalysisObject(
        inputs.experiment_name_baseline,
        irl_path=irl_path,
        experiment_subdirectory=inputs.experiment_subdirectory,
    )
    optimization_data_baseline = analysis_obj_baseline.query_optimization_data(
        excluded_parameters=analysis_obj_baseline.analysis_details.excluded_parameters
    )

    main_analysis_obj = AnalysisObject(
        inputs.main_experiment_name,
        irl_path=irl_path,
        experiment_subdirectory=inputs.experiment_subdirectory,
    )
    main_optimization_data = main_analysis_obj.query_optimization_data(
        excluded_parameters=main_analysis_obj.analysis_details.excluded_parameters
    )

    data_path = subdirectory.joinpath("data/regressions")
    data_path.mkdir(parents=True, exist_ok=True)

    fairy_subset = analysis_obj_baseline.join_optimization_df_and_processed(
        optimization_df=optimization_data_baseline,
        processed_df=analysis_obj_baseline.dfs["individual-variables"],
        variables_of_interest=["DEPTH", "COST", "FAIRY_GOD_CONDITION", "cond"],
    )

    test_subset = analysis_obj_test.join_optimization_df_and_processed(
        optimization_df=optimization_data_test,
        processed_df=analysis_obj_test.dfs["individual-variables"],
        variables_of_interest=["DEPTH", "COST", "FAIRY_GOD_CONDITION", "cond"],
    )

    model_params = list(
        set(main_analysis_obj.cost_details.constant_values)
        - set(main_analysis_obj.analysis_details.excluded_parameters)
    )
    combined = test_subset.merge(
        fairy_subset[model_params + ["trace_pid"]],
        suffixes=("", "_fairy"),
        how="left",
        on="trace_pid",
    )

    for param in model_params:
        logging.info(
            "Descriptive statistics for main experiment (%s) %s",
            inputs.main_experiment_name,
            param,
        )
        logging.info(
            "$M: %.2f, SD: %.2f$",
            main_optimization_data[param].mean(),
            main_optimization_data[param].std(),
        )

        logging.info(
            "Descriptive statistics for validation experiment, test block %s", param
        )
        logging.info(
            "$M: %.2f, SD: %.2f$",
            test_subset[param].mean(),
            test_subset[param].std(),
        )

        logging.info(
            "Descriptive statistics for validation experiment, baseline block %s",
            param,
        )
        logging.info(
            "$M: %.2f, SD: %.2f$",
            fairy_subset[param].mean(),
            fairy_subset[param].std(),
        )

        comparison = pg.mwu(test_subset[param], main_optimization_data[param])
        logging.info(
            "Comparison between validation experiment's test block and "
            "main experiment (%s) for cost parameter: %s",
            inputs.main_experiment_name,
            param,
        )
        logging.info(get_mann_whitney_text(comparison))

        comparison = pg.mwu(fairy_subset[param], main_optimization_data[param])
        logging.info(
            "Comparison between validation experiment's baseline block and "
            "main experiment (%s) for cost parameter: %s",
            inputs.main_experiment_name,
            param,
        )
        logging.info(get_mann_whitney_text(comparison))

        # test is paired, need to have same order of pids, which is why
        # we use combined df
        comparison = pg.wilcoxon(combined[param], combined[f"{param}_fairy"])
        logging.info(
            "Comparison between validation experiment's test and baseline "
            "blocks for cost parameter: %s",
            param,
        )
        logging.info(get_wilcoxon_text(comparison))

    for cost_variable_tuple in [
        ("COST", "given_cost"),
        ("DEPTH", "depth_cost_weight"),
    ]:
        assigned_cost, inferred_cost = cost_variable_tuple
        logging.info("RMSE for %s", inferred_cost)
        logging.info(
            (
                np.sum((combined[assigned_cost] - combined[inferred_cost]) ** 2)
                / len(combined)
            )
            ** (1 / 2)
        )

    loo = StratifiedKFold(n_splits=10)

    sd_values = {
        model_param: main_optimization_data[model_param].std()
        for model_param in model_params
    }

    for cost_variable_tuple in [
        ("COST", "given_cost"),
        ("DEPTH", "depth_cost_weight"),
    ]:
        assigned_cost, inferred_cost = cost_variable_tuple
        combined["prediction_error_{assigned_cost}"] = 0

        for num_split, indices in enumerate(loo.split(combined, combined["cond"])):
            train_index, test_index = indices
            train_data = combined.iloc[train_index]
            test_data = combined.iloc[test_index]

            mod = smf.ols(
                formula=f"{assigned_cost} ~ "
                f"{' + '.join(model_params)}"
                f"+ C(FAIRY_GOD_CONDITION) +  1 + "
                f"{' + '.join([model_param + '_fairy' for model_param in model_params])}",  # noqa: E501
                data=train_data,
            )
            res = mod.fit()

            combined.loc[
                combined["trace_pid"].isin(test_data["trace_pid"].values), "test_fold"
            ] = num_split
            combined.loc[
                combined["trace_pid"].isin(test_data["trace_pid"].values),
                f"prediction_error_{assigned_cost}",
            ] = (
                res.predict(test_data) - test_data[assigned_cost]
            )

        combined[f"prediction_error_squared_{assigned_cost}"] = (
            combined[f"prediction_error_{assigned_cost}"] ** 2
        )
        combined[f"rmse_{assigned_cost}"] = combined[
            f"prediction_error_squared_{assigned_cost}"
        ] ** (1 / 2)

        combined["dummy"] = 1

        test_fold_rmses = (
            combined.groupby(["test_fold"])
            .sum()
            .apply(
                lambda row: np.sqrt(
                    row[f"prediction_error_squared_{assigned_cost}"] / row["dummy"]
                ),
                axis=1,
            )
        )

        logging.info("RMSE for LOO for %s, %s", assigned_cost, inferred_cost)
        logging.info(
            "$M: %.2f, SD: %.2f$",
            test_fold_rmses.mean(),
            test_fold_rmses.std(),
        )

        # fit model to all data, save
        mod = smf.ols(
            formula=f"{assigned_cost} ~"
            f"{' + '.join(model_params)}"
            f" + C(FAIRY_GOD_CONDITION) + 1 + "
            f"{' + '.join([model_param + '_fairy' for model_param in model_params])}",
            data=combined,
        )
        res = mod.fit()
        logging.info(res.summary())
        res.save(data_path.joinpath(f"{assigned_cost}_model.pkl"))
        logging.info("Regression for %s", assigned_cost)
        logging.info("%s", get_regression_text(res))

        combined[f"prediction_error_{assigned_cost}"] = (
            res.predict(combined) - combined[assigned_cost]
        )

        combined[f"prediction_error_squared_{assigned_cost}"] = (
            combined[f"prediction_error_{assigned_cost}"] ** 2
        )
        combined[f"rmse_{assigned_cost}"] = combined[
            f"prediction_error_squared_{assigned_cost}"
        ] ** (1 / 2)

        percentage_under_sd = np.sum(
            combined[f"rmse_{assigned_cost}"] < sd_values[inferred_cost]
        ) / len(combined[f"rmse_{assigned_cost}"])
        logging.info(
            "Number of participants recovered in at least 1 SD for %s: %.2f",
            assigned_cost,
            percentage_under_sd,
        )

    under_both = combined.apply(
        lambda row: np.all(
            [
                row[f"rmse_{assigned_cost}"] < sd_values[inferred_cost]
                for assigned_cost in ["COST", "DEPTH"]
            ]
        ),
        axis=1,
    )
    percentage_under_sd = np.sum(under_both) / len(under_both)
    logging.info(
        "Number of participants recovered in at least 1 SD "
        "of Experiment 1 value for both: %.2f",
        percentage_under_sd,
    )

    all_rmses = combined.melt(value_vars=["rmse_DEPTH", "rmse_COST"])

    plt.figure(figsize=(11.7, 8.27))
    sns.violinplot(x="variable", y="value", data=all_rmses)
    plt.savefig(subdirectory.joinpath("figs/ValidationExperiment_RMSE_violin.png"))

    for cost_variable_tuple in [
        ("COST", "given_cost"),
        ("DEPTH", "depth_cost_weight"),
    ]:
        assigned_cost, inferred_cost = cost_variable_tuple
        logging.info("Test subject difference vs variance {assigned_cost}")
        comparison = pg.ttest(
            combined[f"prediction_error_squared_{assigned_cost}"],
            combined[assigned_cost].var(),
            alternative="less",
        )
        logging.info(get_ttest_text(comparison))
        logging.info(
            "%.3f %.3f",
            combined[f"prediction_error_squared_{assigned_cost}"].mean(),
            combined[assigned_cost].var(),
        )

    for cost_variable_tuple in [
        ("COST", "given_cost"),
        ("DEPTH", "depth_cost_weight"),
    ]:
        assigned_cost, inferred_cost = cost_variable_tuple

        logging.info(
            "Difference in block order for baseline block MAP estimates %s",
            inferred_cost,
        )
        comparison = pg.mwu(
            combined[combined["FAIRY_GOD_CONDITION"]][f"{inferred_cost}_fairy"],
            combined[~combined["FAIRY_GOD_CONDITION"]][f"{inferred_cost}_fairy"],
        )
        logging.info("%s", get_mann_whitney_text(comparison))

        logging.info(
            "Baseline interrupts: %.3f %.3f",
            np.mean(
                combined[combined["FAIRY_GOD_CONDITION"]][f"{inferred_cost}_fairy"]
            ),  # noqa: E501
            np.std(
                combined[combined["FAIRY_GOD_CONDITION"]][f"{inferred_cost}_fairy"]
            ),  # noqa: E501
        )
        logging.info(
            "%.3f %.3f",
            np.mean(
                combined[~combined["FAIRY_GOD_CONDITION"]][f"{inferred_cost}_fairy"]
            ),  # noqa: E501
            np.std(
                combined[~combined["FAIRY_GOD_CONDITION"]][f"{inferred_cost}_fairy"]
            ),  # noqa: E501
        )

        logging.info(
            "Difference in block order for test block MAP estimates %s", inferred_cost
        )
        comparison = pg.mwu(
            combined[combined["FAIRY_GOD_CONDITION"]][inferred_cost],
            combined[~combined["FAIRY_GOD_CONDITION"]][inferred_cost],
        )
        logging.info(get_mann_whitney_text(comparison))

        logging.info(
            "Baseline interrupts: %.3f %.3f",
            np.mean(combined[combined["FAIRY_GOD_CONDITION"]][inferred_cost]),
            np.std(combined[combined["FAIRY_GOD_CONDITION"]][inferred_cost]),
        )
        logging.info(
            "%.3f %.3f",
            np.mean(combined[~combined["FAIRY_GOD_CONDITION"]][inferred_cost]),
            np.std(combined[~combined["FAIRY_GOD_CONDITION"]][inferred_cost]),
        )
