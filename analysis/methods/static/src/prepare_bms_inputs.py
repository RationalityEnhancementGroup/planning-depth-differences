"""Prepare data for Bayesian Model Comparison (in spm_BMS_for_me.m)."""
import sys
from pathlib import Path

from costometer.utils import standard_parse_args

if __name__ == "__main__":
    """
    Example usage:
    python src/plot_bms.py -e MainExperiment
    """
    irl_path = Path(__file__).resolve().parents[4]
    analysis_obj, inputs, subdirectory = standard_parse_args(
        description=sys.modules[__name__].__doc__,
        irl_path=irl_path,
        filename=Path(__file__).stem,
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
