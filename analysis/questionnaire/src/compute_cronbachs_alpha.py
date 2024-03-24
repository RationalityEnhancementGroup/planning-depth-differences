import logging
import sys
from pathlib import Path

import pandas as pd
import pingouin as pg
import yaml
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

    with open(
        subdirectory.joinpath(f"inputs/yamls/{inputs.experiment_name}.yaml"), "r"
    ) as stream:
        experiment_arguments = yaml.safe_load(stream)

    individual_items = pd.concat(
        [
            pd.read_csv(
                irl_path.joinpath(f"data/processed/{session}/individual_items.csv")
            )
            for session in experiment_arguments["sessions"]
        ]
    )
    individual_items = individual_items[individual_items.columns.difference(["gender"])]
    individual_items = individual_items.set_index("pid")

    questionnaire_mapping = {
        "anxiety": "Anxiety",
        "bit": "Brief Inventory of Thriving",
        "cfc": "Consideration of Future Consequences",
        "dts": "Distress Tolerance Scale",
        "fos": "Future Orientation Scale",
        "ius": "Intolerance of Uncertainty",
        "lifreg": "Life Regrets",
        "ptp": "Propensity to Plan",
        "reg": "Propensity to Regret",
        "satis": "Satisfaction",
        "uppsp": "UPPSP",
        "apathy": "apathy",
        "bis": "bis",
        "dospert.": "dospert.",
        "dospert-eb.": "dospert-eb.",
        "dospert-rp.": "dospert-rp.",
        "eat": "eat",
        "leb": "leb",
        "ocir": "ocir",
        "zung": "zung",
    }

    for question_id in questionnaire_mapping.keys():
        matching_columns = [
            col for col in individual_items.columns if col.startswith(question_id)
        ]
        res, ci = pg.cronbach_alpha(
            individual_items[matching_columns], nan_policy="listwise"
        )
        logging.info("{question_id}, {len(matching_columns)}, {res:0.3f}, {ci}")
