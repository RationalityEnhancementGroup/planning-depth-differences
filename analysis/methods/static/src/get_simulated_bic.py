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
    data_path = irl_path.joinpath(f"analysis/{inputs.experiment_subdirectory}")

    data_path.joinpath("log").mkdir(parents=True, exist_ok=True)
    data_path.joinpath("data").mkdir(parents=True, exist_ok=True)

    bic_dict = {}
    for experiment_name in inputs.experiment_name.split(","):
        analysis_obj = AnalysisObject(
            experiment_name,
            irl_path=irl_path,
            experiment_subdirectory=inputs.experiment_subdirectory,
        )

        optimization_data = analysis_obj.query_optimization_data(
            excluded_parameters=analysis_obj.excluded_parameters
        )

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

    with open(data_path.joinpath(f"data/{inputs.experiment_name}.pickle"), "wb") as f:
        pickle.dump(bic_dict, f)
