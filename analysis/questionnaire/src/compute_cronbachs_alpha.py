from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import pingouin as pg
import yaml

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp",
        dest="experiment_name",
    )
    inputs = parser.parse_args()

    data_path = Path(__file__).resolve().parents[1]
    irl_path = Path(__file__).resolve().parents[3]

    with open(
        data_path.joinpath(f"inputs/yamls/{inputs.experiment_name}.yaml"), "r"
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

    individual_items

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
        # 'alcohol':'alcohol',
        "apathy": "apathy",
        "bis": "bis",
        "dospert": "dospert",
        "dospert-eb": "dospert-eb",
        "dospert-rp": "dospert-rp",
        "eat": "eat",
        "leb": "leb",
        "ocir": "ocir",
        "zung": "zung",
    }

    for question_id, questionnaire_name in questionnaire_mapping.items():
        matching_columns = [
            col for col in individual_items.columns if col.startswith(question_id)
        ]
        print(
            question_id,
            len(matching_columns),
            pg.cronbach_alpha(
                individual_items[matching_columns], nan_policy="listwise"
            ),
        )

# def cronbach_to_words(alpha):
#     if alpha > .9:
#         return "excellent"
#     elif alpha < .7:
#         return "poor"
#     else:
#         return "acceptable"
#
# def print_cronbachs_alpha(res, question_name, num_questions):
#     alpha, range = res
#     print(f"The overall reliability of the overall {question_name} scale was
#     {cronbach_to_words(alpha)} (${}$ items; $alpha={alpha:.2f}$, 95% C.I.: [${range[0]:.2f}$,${range[1]:.2f}$]).")
#     print(f"The {} subscale had {cronbach_to_words()} reliability (${}$ items;
#     $alpha={alpha:.2f}$, 95% C.I.: [${range[0]:.2f}$,${range[1]:.2f}$]).")
