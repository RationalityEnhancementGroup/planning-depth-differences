"""Prepare data for Bayesian Model Comparison (in spm_BMS_for_me.m)."""

from argparse import ArgumentParser
from pathlib import Path

from costometer.utils import AnalysisObject

if __name__ == "__main__":
    """
    Example usage:
    python src/plot_bms.py -e MainExperiment
    """
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp",
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

    analysis_obj = AnalysisObject(
        inputs.experiment_name,
        irl_path=irl_path,
        experiment_subdirectory=inputs.experiment_subdirectory,
    )

    optimization_data = analysis_obj.query_optimization_data()
    optimization_data = optimization_data[
        optimization_data.apply(
            lambda row: set(analysis_obj.analysis_details.excluded_parameters).issubset(
                row["model"]
            )
            or (row["Model Name"] == "Null"),
            axis=1,
        )
    ]

    irl_path.joinpath("data/bms/inputs/").mkdir(parents=True, exist_ok=True)
    irl_path.joinpath("data/bms/outputs/").mkdir(parents=True, exist_ok=True)

    if not irl_path.joinpath(f"data/bms/inputs/{inputs.experiment_name}.csv").is_file():
        pivoted_df = optimization_data.pivot(
            index="trace_pid", columns="Model Name", values="bic"
        )
        pivoted_df = pivoted_df.apply(lambda evidence: -0.5 * evidence)
        pivoted_df.to_csv(
            irl_path.joinpath(f"data/bms/inputs/{inputs.experiment_name}.csv")
        )
