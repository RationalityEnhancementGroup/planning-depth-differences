from argparse import ArgumentParser
from pathlib import Path

import dill as pickle
from costometer.utils.analysis_utils import AnalysisObject

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-e", "--exp", dest="experiment_name", default="MainExperiment")
    parser.add_argument(
        "-s",
        "--subdirectory",
        default="methods/static",
        dest="experiment_subdirectory",
        metavar="experiment_subdirectory",
    )
    inputs = parser.parse_args()

    irl_path = Path(__file__).parents[2]

    analysis_obj = AnalysisObject(
        inputs.experiment_name,
        irl_path=irl_path,
        experiment_subdirectory=inputs.experiment_subdirectory,
    )

    model_params = list(analysis_obj.cost_details["constant_values"])

    # random effects
    optimization_data = analysis_obj.get_random_effects_optimization_data()

    deduped_parameter_values = optimization_data[
        model_params + ["noise"]
    ].drop_duplicates()
    possible_parameters = deduped_parameter_values.drop_duplicates().to_dict("records")

    with open(
        irl_path.joinpath(
            "cluster/parameters/simulations/participants_random_effects.pkl"
        ),
        "wb",
    ) as f:
        pickle.dump(possible_parameters, f)

    for excluded_parameters in analysis_obj.trial_by_trial_models:
        optimization_data = analysis_obj.query_optimization_data(
            excluded_parameters=excluded_parameters
        )

        # in case not alphabetically sorted
        model_text = ",".join(sorted(excluded_parameters.split(",")))

        deduped_parameter_values = optimization_data[
            model_params + ["noise"]
        ].drop_duplicates()
        possible_parameters = deduped_parameter_values.drop_duplicates().to_dict(
            "records"
        )

        with open(
            irl_path.joinpath(
                f"cluster/parameters/simulations/"
                f"participants"
                f"{'_' + model_text if model_text != '' else model_text}.pkl"
            ),
            "wb",
        ) as f:
            pickle.dump(possible_parameters, f)
