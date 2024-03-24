from argparse import ArgumentParser
from pathlib import Path

import dill as pickle
from costometer.utils import AnalysisObject
from statsmodels.tools.eval_measures import bic

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp",
        default="OptimalBIC,SimulatedParticipant,SoftmaxRecovery",
        dest="experiment_name",
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

    bic_dict = {}
    for experiment_name in inputs.experiment_name.split(","):
        analysis_obj = AnalysisObject(
            experiment_name,
            irl_path=irl_path,
            experiment_subdirectory=inputs.experiment_subdirectory,
        )

        optimization_data = analysis_obj.query_optimization_data(
            excluded_parameters=analysis_obj.analysis_details.excluded_parameters
        )

        # needed for SimulatedParticipant since some cost parameters
        # are shared by multiple participants have
        if experiment_name == "SimulatedParticipant":
            main_analysis_obj = AnalysisObject(
                "MainExperiment",
                irl_path=irl_path,
                experiment_subdirectory=inputs.experiment_subdirectory,
            )
            main_optimization_data = main_analysis_obj.query_optimization_data(
                excluded_parameters=analysis_obj.analysis_details.excluded_parameters
            )[list(main_analysis_obj.cost_details.constant_values)]

            optimization_data = optimization_data.merge(
                main_optimization_data,
                left_on=[
                    f"sim_{param}"
                    for param in main_analysis_obj.cost_details.constant_values
                ],
                right_on=list(main_analysis_obj.cost_details.constant_values),
                how="inner",
            )
            assert len(optimization_data) == len(main_optimization_data)
            assert not optimization_data["mle"].isnull().any()
            assert not optimization_data["num_clicks"].isnull().any()

            bic_df = (
                optimization_data.groupby(["Model Name", "Number Parameters"])
                .sum()
                .reset_index()
            )

            bic_val = sum(
                bic_df.apply(
                    lambda row: bic(
                        llf=row["mle"],
                        nobs=row["num_clicks"],
                        df_modelwc=row["Number Parameters"],
                    ),
                    axis=1,
                )
            )

            bic_dict[experiment_name] = bic_val

        with open(subdirectory.joinpath("data/Simulated_BIC.pickle"), "wb") as f:
            pickle.dump(bic_dict, f)
