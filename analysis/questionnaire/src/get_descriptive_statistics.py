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
        "Task motivation: $%0.3f$ (SD: $%0.3f$)",
        task_motivation.mean(),
        task_motivation.std(),
    )

    logging.info("===========")

    # -1 means we don't have data, 4 means they weren't sure
    overall_effort = analysis_obj.dfs["quiz-and-demo"]["effort"]
    overall_effort[pd.isnull(overall_effort)] = -1

    logging.info(
        "Overall effort: $%0.3f$ (SD: $%0.3f$)",
        overall_effort[~overall_effort.isin([-1, 4])].mean(),
        overall_effort[~overall_effort.isin([-1, 4])].std(),
    )
    logging.info("No response: %d", len(overall_effort[overall_effort == -1]))
    logging.info("Uncertain: %d", len(overall_effort[overall_effort == 4]))
    logging.info(Counter(overall_effort))

    logging.info("===========")

    age = analysis_obj.dfs["quiz-and-demo"]["age"]
    gender = analysis_obj.dfs["quiz-and-demo"]["gender"]

    logging.info(
        "The final sample of $%d$ participants consisted of $%d$ women, "
        "$%d$ men and $%d$ people who were non-binary or did not provide their gender.",
        len(gender),
        len(gender[gender == "female"]),
        len(gender[gender == "male"]),
        len(gender[~gender.isin(["male", "female"])]),
    )
    logging.info(
        "The median age of participants was $%.2f$ (range $%d$ to $%d$).",
        median(age[~pd.isnull(age)]),
        min(age[~pd.isnull(age)]),
        max(age[~pd.isnull(age)]),
    )
    logging.info("Age data was not provided by $%d$ participants.", sum(pd.isnull(age)))
    logging.info("===========")
