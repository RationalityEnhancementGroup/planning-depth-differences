from pathlib import Path

import dill as pickle
import pandas as pd

if __name__ == "__main__":
    irl_path = Path(__file__).parents[2]

    possible_parameters = [
        {
            "back_added_cost": 0,
            "forw_added_cost": 0,
            "distance_multiplier": 0,
            "depth_cost_weight": 0,
            "given_cost": 1,
            "kappa": 1,
            "gamma": 1,
        }
    ] * 122

    possible_parameters = pd.DataFrame(possible_parameters).to_dict("records")

    with open(
        irl_path.joinpath("cluster/parameters/simulations/no_added_cost.pkl"),
        "wb",
    ) as f:
        pickle.dump(possible_parameters, f)
