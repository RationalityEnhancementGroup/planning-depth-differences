from pathlib import Path
import dill as pickle
from costometer.utils.analysis_utils import AnalysisObject
import pandas as pd

if __name__ == "__main__":
    irl_path = Path(__file__).parents[2]

    back_added_cost_values = [0]
    forw_add_cost_values = [0, -2.5, -10]
    distance_multiplier_values = [0, 2.5, 10]
    depth_cost_values = [10, 5, 0, -5]
    given_cost_values = [10, 5, 0, -5]

    with open(
        irl_path.joinpath(f"cluster/parameters/gammas/partial.txt"), "r"
    ) as f:
        gamma_values = [float(val) for val in f.read().splitlines()]


    with open(
        irl_path.joinpath(f"cluster/parameters/kappas/partial.txt"), "r"
    ) as f:
        kappa_values = [float(val) for val in f.read().splitlines()]

    temps = [1]
    noises = [0]

    possible_parameters_df = []
    for back_added_cost in back_added_cost_values:
        for forw_added_cost in forw_add_cost_values:
            for distance_multiplier in distance_multiplier_values:
                for depth_cost_weight in depth_cost_values:
                    for given_cost in given_cost_values:
                        cost_parameters = {"back_added_cost" : back_added_cost,
                                           "forw_added_cost": forw_added_cost,
                                           "distance_multiplier" : distance_multiplier,
                                           "depth_cost_weight" : depth_cost_weight,
                                           "given_cost" : given_cost}

                        for temp in temps:
                            for noise in noises:
                                policy_parameters = {"temp": temp, "noise": noise}
                                for gamma in gamma_values:
                                    for kappa in kappa_values:
                                        possible_parameters_df.append({"back_added_cost" : back_added_cost,
                                           "forw_added_cost": forw_added_cost,
                                           "distance_multiplier" : distance_multiplier,
                                           "depth_cost_weight" : depth_cost_weight,
                                           "given_cost" : given_cost,"kappa":kappa, "gamma": gamma,"temp": temp, "noise": noise})

    possible_parameters = pd.DataFrame(possible_parameters_df).drop_duplicates().to_dict("records")

    with open(irl_path.joinpath("cluster/parameters/simulations/reduced.pkl"), "wb") as f:
        pickle.dump(possible_parameters, f)

    analysis_obj = AnalysisObject(
        "MainExperiment",
        irl_path=irl_path,
        experiment_subdirectory="methods/static",
    )

    optimization_data = analysis_obj.query_optimization_data()
    deduped_parameter_values = optimization_data[optimization_data["applied_policy"]=="SoftmaxPolicy"][["back_added_cost","forw_added_cost","distance_multiplier","depth_cost_weight","given_cost", "temp", "noise", "kappa", "gamma"]].drop_duplicates()

    possible_parameters = deduped_parameter_values.drop_duplicates().to_dict("records")

    with open(irl_path.joinpath("cluster/parameters/simulations/participants.pkl"), "wb") as f:
        pickle.dump(possible_parameters, f)


    possible_parameters_df = []
    for back_added_cost in [0]:
        for forw_added_cost in [0]:
            for distance_multiplier in [0]:
                for depth_cost_weight in [0]:
                    for given_cost in [1]:
                        cost_parameters = {"back_added_cost" : back_added_cost,
                                           "forw_added_cost": forw_added_cost,
                                           "distance_multiplier" : distance_multiplier,
                                           "depth_cost_weight" : depth_cost_weight,
                                           "given_cost" : given_cost}

                        for temp in [1]:
                            for noise in [0]:
                                policy_parameters = {"temp": temp, "noise": noise}
                                for gamma in gamma_values:
                                    for kappa in [1]:
                                        possible_parameters_df.append({"back_added_cost" : back_added_cost,
                                           "forw_added_cost": forw_added_cost,
                                           "distance_multiplier" : distance_multiplier,
                                           "depth_cost_weight" : depth_cost_weight,
                                           "given_cost" : given_cost,"kappa":kappa, "gamma": gamma,"temp": temp, "noise": noise})


    possible_parameters = pd.DataFrame(possible_parameters_df).drop_duplicates().to_dict("records")

    with open(irl_path.joinpath("cluster/parameters/simulations/gamma_only.pkl"), "wb") as f:
        pickle.dump(possible_parameters, f)



    possible_parameters_df = []
    for back_added_cost in [0]:
        for forw_added_cost in [0]:
            for distance_multiplier in [0]:
                for depth_cost_weight in [0]:
                    for given_cost in [1]:
                        cost_parameters = {"back_added_cost" : back_added_cost,
                                           "forw_added_cost": forw_added_cost,
                                           "distance_multiplier" : distance_multiplier,
                                           "depth_cost_weight" : depth_cost_weight,
                                           "given_cost" : given_cost}

                        for temp in [1]:
                            for noise in [0]:
                                policy_parameters = {"temp": temp, "noise": noise}
                                for gamma in [1]:
                                    for kappa in kappa_values:
                                        possible_parameters_df.append({"back_added_cost" : back_added_cost,
                                           "forw_added_cost": forw_added_cost,
                                           "distance_multiplier" : distance_multiplier,
                                           "depth_cost_weight" : depth_cost_weight,
                                           "given_cost" : given_cost,"kappa":kappa, "gamma": gamma,"temp": temp, "noise": noise})


    possible_parameters = pd.DataFrame(possible_parameters_df).drop_duplicates().to_dict("records")

    with open(irl_path.joinpath("cluster/parameters/simulations/kappa_only.pkl"), "wb") as f:
        pickle.dump(possible_parameters, f)