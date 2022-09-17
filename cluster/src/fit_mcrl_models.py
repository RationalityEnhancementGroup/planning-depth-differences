import ast
import sys
from pathlib import Path

from mcl_toolbox.fit_mcrl_models import fit_model

if __name__ == "__main__":
    exp_name = sys.argv[1]
    model_index = int(sys.argv[2])
    optimization_criterion = sys.argv[3]
    pid = int(sys.argv[4])
    number_of_trials = int(sys.argv[5])
    # other_params = {"plotting": True}
    other_params = {}
    if len(sys.argv) > 6:
        other_params = ast.literal_eval(sys.argv[6])
    else:
        other_params = {}

    # set data path
    data_path = Path(__file__).parents[2].joinpath("data/processed/")
    save_path = Path(__file__).parents[1].joinpath("data/mcrl")

    if "exp_attributes" not in other_params:
        exp_attributes = {
            "exclude_trials": None,  # Trials to be excluded
            "block": None,  # Block of the experiment
            "experiment": None,
            # Experiment object can be passed directly
            # with pipeline and normalized features attached
            "click_cost": 1,
        }
        other_params["exp_attributes"] = exp_attributes

    if "optimization_params" not in other_params:
        optimization_params = {
            "optimizer": "hyperopt",
            "num_simulations": 1,
            "max_evals": 1,
        }
        other_params["optimization_params"] = optimization_params

    fit_model(
        exp_name=exp_name,
        pid=pid,
        number_of_trials=number_of_trials,
        data_path=data_path,
        save_path=save_path,
        model_index=model_index,
        optimization_criterion=optimization_criterion,
        **other_params,
    )
