from pathlib import Path

import blosc
import dill as pickle
import numpy as np
import pandas as pd
from costometer.utils import get_param_string

path = Path(__file__).parents[0].joinpath("cluster/data")

experiment_setting = "small_test_case"
cost_function_name = "distance_graph_cost"
bmps_weights = []
print(
    path.joinpath(
        f"bmps/{experiment_setting}/{cost_function_name}/"
        f"BMPS_{experiment_setting}_*.dat"  # noqa: E501
    )
)
filenames = path.glob(
    f"bmps/{experiment_setting}/{cost_function_name}/"
    f"BMPS_{experiment_setting}_*.dat"  # noqa: E501
)
for filename in filenames:
    with open(filename, "rb") as f:
        compressed_data = f.read()

    decompressed_data = blosc.decompress(compressed_data)
    info = pickle.loads(decompressed_data)
    bmps_weights.append(
        {
            **{f"W{w_idx}": w for w_idx, w in enumerate(info["weights"])},
            **info["cost_params"],
        }
    )

bmps_weights

bmps_weights_df = pd.DataFrame.from_dict(bmps_weights)
