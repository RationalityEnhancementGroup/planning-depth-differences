from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from statistics import median

import pandas as pd
from costometer.utils import AnalysisObject

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp",
        dest="experiment_name",
    )
    parser.add_argument(
        "-s",
        "--subdirectory",
        default="questionnaire",
        dest="experiment_subdirectory",
        metavar="experiment_subdirectory",
    )
    inputs = parser.parse_args()

    data_path = Path(__file__).resolve().parents[1]
    irl_path = Path(__file__).resolve().parents[3]

    analysis_obj = AnalysisObject(
        inputs.experiment_name,
        irl_path=irl_path,
        experiment_subdirectory=inputs.experiment_subdirectory,
    )

    task_motivation = analysis_obj.dfs["quiz-and-demo"]["mouselab-quiz-post_Q5"]
    print(
        f"Task motivation: ${task_motivation.mean():0.3f}$"
        f" (SD: ${task_motivation.std():0.3f}$)"
    )

    print("===========")

    # -1 means we don't have data, 4 means they weren't sure
    overall_effort = analysis_obj.dfs["quiz-and-demo"]["effort"]
    overall_effort[pd.isnull(overall_effort)] = -1

    print(
        f"Overall effort: ${overall_effort[~overall_effort.isin([-1,4])].mean():0.3f}$"
        f" (SD: ${overall_effort[~overall_effort.isin([-1,4])].std():0.3f}$)"
    )
    print(f"No response: {len(overall_effort[overall_effort == -1])}")
    print(f"Uncertain: {len(overall_effort[overall_effort == 4])}")
    print(Counter(overall_effort))

    print("===========")

    age = analysis_obj.dfs["quiz-and-demo"]["age"]
    gender = analysis_obj.dfs["quiz-and-demo"]["gender"]

    print(
        f"The final sample of ${len(gender)}$ participants consisted of"
        f" ${len(gender[gender == 'female'])}$ women, ${len(gender[gender == 'male'])}$"
        f" men and ${len(gender[~gender.isin(['male','female'])])}$ people who were "
        f"non-binary or did not provide their gender."
    )
    print(
        f"The median age of participants was ${median(age[~pd.isnull(age)])}$"
        f" (range ${min(age[~pd.isnull(age)])}$ to ${max(age[~pd.isnull(age)])}$)."
    )
    print(f"Age data was not provided by ${sum(pd.isnull(age))}$ participants.")
    print("===========")
