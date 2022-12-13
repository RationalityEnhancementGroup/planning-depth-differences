import glob
from pathlib import Path

import dill as pickle
import pandas as pd

data_path = Path(__file__).resolve().parents[1]
data_path.joinpath("OSF").mkdir(parents=True, exist_ok=True)

# First load all data
sessions = ["0.0.3", "0.0.4", "0.0.5"]

# load data
full_data = {}

# read in sessions and concatenate
for run in sessions:
    for file_path in data_path.glob(f"raw/{run}/*.csv"):
        # don't want to save identifiable bonuses
        # file, information is already in data
        if "bonuses" not in str(file_path):
            file_name = file_path.stem
            curr_data_frame = pd.read_csv(file_path)
            curr_data_frame["run"] = run

            if file_name not in full_data:
                full_data[file_name] = [curr_data_frame]
            else:
                full_data[file_name].append(curr_data_frame)

full_data = {k: pd.concat(v) for k, v in full_data.items()}

# delete dates / possible location info in general info
del full_data["question_data"]["startTime"]
del full_data["general_info"]["beginhit"]
del full_data["general_info"]["beginexp"]
del full_data["general_info"]["endhit"]
del full_data["general_info"]["language"]

# delete free text responses for now
del full_data["survey-text"]["response"]

# save dfs
for trial_type, df in full_data.items():
    df.to_csv(data_path.joinpath(f"OSF/{trial_type}.csv"))

# load in demographics to get approved pids
all_demos = pd.concat([pd.read_csv(f) for f in glob.glob("../raw/csv/*/prolific*.csv")])
all_demos = all_demos[all_demos["Status"] == "APPROVED"]

prolific_demo = all_demos[["Participant id", "Age", "Sex"]].reset_index()

# load mapping of worker ids > ids
with open(data_path.joinpath("mturk_id_mapping.pickle"), "rb") as f:
    pid_labeler_data = pickle.load(f)

# pid_labeler_data is str(bytes(prolific id)), also some approved at least one user has no data (???) # noqa : E501
prolific_demo["Participant id"] = prolific_demo["Participant id"].apply(
    lambda pid: "NO_DATA"
    if pid.encode() not in pid_labeler_data
    else pid_labeler_data[pid.encode()]
)

prolific_demo.to_csv(data_path.joinpath("OSF/prolific_demographics.csv"))

# now check bonuses
all_bonuses = pd.concat(
    [pd.read_csv(f"../raw/{run}/bonuses.csv") for run in ["0.0.3", "0.0.4", "0.0.5"]]
)
# workerid are of form "b'vals_workerid'" so transform
all_bonuses["workerid"] = all_bonuses["workerid"].apply(lambda st: st[2:-1])
# for earlier runs, there are some duplicates of workerid & calculated_bonus
all_bonuses = all_bonuses[["workerid", "calculated_bonus"]].drop_duplicates()

# 1's folder contains 1a and 1b; 2a had no participants
for experiment in ["1", "2", "2b", "3", "4", "5"]:
    # now read demographic files only for a certain run
    all_demos = pd.concat(
        [pd.read_csv(f) for f in glob.glob(f"../raw/csv/{experiment}/prolific*.csv")]
    )
    approved = all_demos[all_demos["Status"] == "APPROVED"]

    paid_bonuses = all_bonuses[all_bonuses["workerid"].isin(approved["Participant id"])]

    # only sum bonuses for participants with nonnegative bonuses
    print(
        experiment,
        paid_bonuses[paid_bonuses["calculated_bonus"] > 0]["calculated_bonus"].sum(),
    )
