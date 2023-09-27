import io
import json
from argparse import ArgumentParser
from pathlib import Path

import dill as pickle
import pandas as pd

if __name__ == "__main__":
    """
    Using code and modified excel from Toby Wise: https://osf.io/b95w2/
    """
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--column",
        dest="column",
        help="Experiment column to subset on",
        default=None,
    )
    parser.add_argument("-f", "--file_name", dest="file_name", default=None)

    args = parser.parse_args()

    questionnaire_files_path = (
        Path(__file__).parents[1].joinpath("inputs/questionnaire_files/")
    )
    questionnaire_data = pd.read_csv(
        questionnaire_files_path.joinpath("questionnaires_info.csv"),
        encoding="ISO-8859-1",
    )

    # Build key
    solutions = {measure: {} for measure in questionnaire_data["Measure"].unique()}

    for row_idx, row in questionnaire_data.iterrows():
        options = [ans.strip() for ans in row["Options"].split(";")]
        row_id = row["id"].replace("”", "").replace("“", "")
        if ";" in row["Scoring"]:
            scores = [int(score.strip()) for score in row["Scoring"].split(";")]
            solutions[row["Measure"]][row_id] = {
                ans_idx: scores[ans_idx] for ans_idx, option in enumerate(options)
            }
        else:
            try:  # if this doesn't work, the answer is a string
                solutions[row["Measure"]][row_id] = float(row["Scoring"])
            except ValueError:
                solutions[row["Measure"]][row_id] = row["Scoring"]

    pickle.dump(
        solutions,
        open(questionnaire_files_path.joinpath(f"solutions_{args.column}.pkl"), "wb"),
    )

    if args.column:
        questionnaire_data = questionnaire_data[questionnaire_data[args.column] == 1]

    data = {}

    for measure in questionnaire_data["Measure"].unique():
        data[measure] = dict(
            questions=[],
            preamble=questionnaire_data["Preamble"][
                questionnaire_data["Measure"] == measure
            ].iloc[0],
            name=measure,
        )
        for item in questionnaire_data[
            questionnaire_data["Measure"] == measure
        ].iterrows():
            data[measure]["questions"].append(
                dict(
                    prompt=item[1].Question,
                    labels=[ans.strip() for ans in item[1]["Options"].split(";")],
                    question_id=item[1].id,
                    reverse_coded=item[1].ReverseCoded,
                )
            )

    json_data = json.dumps(data, ensure_ascii=False)

    if not args.file_name:
        if args.column:
            args.file_name = f"questionnaire_{args.column}"
        else:
            args.file_name = "questionnaire"

    with io.open(
        questionnaire_files_path.joinpath(f"{args.file_name}.txt"), "w", encoding="utf8"
    ) as outfile:
        outfile.write(str(json_data))
