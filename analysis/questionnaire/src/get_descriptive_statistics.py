import logging
import sys
from collections import Counter
from pathlib import Path
from statistics import median

import pandas as pd
from costometer.utils import standard_parse_args

if __name__ == "__main__":
    irl_path = Path(__file__).resolve().parents[3]
    analysis_obj, inputs, subdirectory = standard_parse_args(
        description=sys.modules[__name__].__doc__,
        irl_path=irl_path,
        filename=Path(__file__).stem,
        default_experiment="QuestMain",
        default_subdirectory="questionnaire",
    )

    task_motivation = analysis_obj.dfs["quiz-and-demo"]["mouselab-quiz-post_Q5"]
    logging.info(
        f"Task motivation: ${task_motivation.mean():0.3f}$"
        f" (SD: ${task_motivation.std():0.3f}$)"
    )

    logging.info("===========")

    # -1 means we don't have data, 4 means they weren't sure
    overall_effort = analysis_obj.dfs["quiz-and-demo"]["effort"]
    overall_effort[pd.isnull(overall_effort)] = -1

    logging.info(
        f"Overall effort: ${overall_effort[~overall_effort.isin([-1,4])].mean():0.3f}$"
        f" (SD: ${overall_effort[~overall_effort.isin([-1,4])].std():0.3f}$)"
    )
    logging.info("No response: {len(overall_effort[overall_effort == -1])}")
    logging.info("Uncertain: {len(overall_effort[overall_effort == 4])}")
    logging.info(Counter(overall_effort))

    logging.info("===========")

    age = analysis_obj.dfs["quiz-and-demo"]["age"]
    gender = analysis_obj.dfs["quiz-and-demo"]["gender"]

    logging.info(
        f"The final sample of ${len(gender)}$ participants consisted of"
        f" ${len(gender[gender == 'female'])}$ women, ${len(gender[gender == 'male'])}$"
        f" men and ${len(gender[~gender.isin(['male','female'])])}$ people who were "
        f"non-binary or did not provide their gender."
    )
    logging.info(
        f"The median age of participants was ${median(age[~pd.isnull(age)])}$"
        f" (range ${min(age[~pd.isnull(age)])}$ to ${max(age[~pd.isnull(age)])}$)."
    )
    logging.info("Age data was not provided by ${sum(pd.isnull(age))}$ participants.")
    logging.info("===========")
