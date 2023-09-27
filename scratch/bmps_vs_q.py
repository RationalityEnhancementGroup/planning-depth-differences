from pathlib import Path

import blosc
import dill as pickle
import numpy as np
from costometer.utils import get_param_string

path = Path(__file__).parents[0].joinpath("cluster/data")

experiment_setting = "small_test_case"
cost_function_name = "distance_graph_cost"
bmps_q = {}
print(
    path.joinpath(
        f"bmps/preferences/{experiment_setting}/{cost_function_name}/"
        f"BMPS_{experiment_setting}_*.dat"  # noqa: E501
    )
)
filenames = path.glob(
    f"bmps/preferences/{experiment_setting}/{cost_function_name}/"
    f"BMPS_{experiment_setting}_*.dat"  # noqa: E501
)
for filename in filenames:
    with open(filename, "rb") as f:
        compressed_data = f.read()

    decompressed_data = blosc.decompress(compressed_data)
    info = pickle.loads(decompressed_data)
    bmps_q[get_param_string(info["cost_params"])] = info["q_dictionary"]

q_q = {}
print(
    path.joinpath(
        f"q_files/{experiment_setting}/{cost_function_name}/"
        f"Q{experiment_setting}_*.dat"  # noqa: E501
    )
)
filenames = path.glob(
    f"q_files/{experiment_setting}/{cost_function_name}/"
    f"Q_{experiment_setting}_*.dat"  # noqa: E501
)
for filename in filenames:
    with open(filename, "rb") as f:
        compressed_data = f.read()

    decompressed_data = blosc.decompress(compressed_data)
    info = pickle.loads(decompressed_data)
    # print(info["cost_params"], get_param_string(info["cost_params"]))
    q_q[get_param_string(info["cost_params"])] = info["q_dictionary"]


for key in bmps_q.keys():
    curr_bmps_q = bmps_q[key]
    curr_q_q = q_q[key]

    for sa_pair in curr_q_q.keys():
        if np.abs(curr_bmps_q[sa_pair] - curr_q_q[sa_pair]) > 0.00001:
            print(key)
            print(sa_pair)
            print((curr_bmps_q[sa_pair], curr_q_q[sa_pair]))

            filenames = path.glob(
                f"bmps/{experiment_setting}/{cost_function_name}/"
                f"BMPS_{experiment_setting}_*{key}*.dat"  # noqa: E501
            )
            for filename in filenames:
                with open(filename, "rb") as f:
                    compressed_data = f.read()

                decompressed_data = blosc.decompress(compressed_data)
                info = pickle.loads(decompressed_data)
                print(info)
