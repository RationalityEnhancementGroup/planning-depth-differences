from pathlib import Path

import dill as pickle
import pandas as pd

if __name__ == "__main__":
    irl_path = Path(__file__).parents[2]

    back_added_cost_values = [0, 5, 10]
    forw_add_cost_values = [0, 5, 10]
    distance_multiplier_values = [0, 5, 10]
    depth_cost_values = [10, 5, 0]
    given_cost_values = [10, 5, 0]

    gamma_values = [1]
    kappa_values = [1]

    temps = [0.1, 1, 10]
    noises = [0]

    possible_parameters_df = []
    for back_added_cost in back_added_cost_values:
        for forw_added_cost in forw_add_cost_values:
            for distance_multiplier in distance_multiplier_values:
                for depth_cost_weight in depth_cost_values:
                    for given_cost in given_cost_values:
                        cost_parameters = {
                            "back_added_cost": back_added_cost,
                            "forw_added_cost": forw_added_cost,
                            "distance_multiplier": distance_multiplier,
                            "depth_cost_weight": depth_cost_weight,
                            "given_cost": given_cost,
                        }

                        for temp in temps:
                            for noise in noises:
                                policy_parameters = {"temp": temp, "noise": noise}
                                for gamma in gamma_values:
                                    for kappa in kappa_values:
                                        possible_parameters_df.append(
                                            {
                                                "back_added_cost": back_added_cost,
                                                "forw_added_cost": forw_added_cost,
                                                "distance_multiplier": distance_multiplier,  # noqa: E501
                                                "depth_cost_weight": depth_cost_weight,  # noqa: E501
                                                "given_cost": given_cost,
                                                "kappa": kappa,
                                                "gamma": gamma,
                                                "temp": temp,
                                                "noise": noise,
                                            }
                                        )

    possible_parameters = (
        pd.DataFrame(possible_parameters_df).drop_duplicates().to_dict("records")
    )

    with open(
        irl_path.joinpath("cluster/parameters/simulations/reduced.pkl"),
        "wb",
    ) as f:
        pickle.dump(possible_parameters, f)
