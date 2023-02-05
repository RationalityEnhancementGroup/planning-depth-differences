from pathlib import Path

import dill as pickle
from costometer.utils.analysis_utils import AnalysisObject

if __name__ == "__main__":
    irl_path = Path(__file__).parents[2]

    analysis_obj = AnalysisObject(
        "MainExperiment",
        irl_path=irl_path,
        experiment_subdirectory="methods/static",
    )

    optimization_data = analysis_obj.query_optimization_data(
        excluded_parameters=analysis_obj.excluded_parameters
    )
    optimization_data = optimization_data[
        optimization_data["applied_policy"] == "SoftmaxPolicy"
    ]

    # in case not alphabetically sorted
    model_text = ",".join(sorted(analysis_obj.excluded_parameters.split(",")))
    model_params = list(analysis_obj.cost_details["constant_values"])

    deduped_parameter_values = optimization_data[
        model_params + ["noise"]
    ].drop_duplicates()
    possible_parameters = deduped_parameter_values.drop_duplicates().to_dict("records")

    with open(
        irl_path.joinpath(
            f"cluster/parameters/simulations/"
            f"participants{'_' + model_text if model_text != '' else model_text}.pkl"
        ),
        "wb",
    ) as f:
        pickle.dump(possible_parameters, f)
