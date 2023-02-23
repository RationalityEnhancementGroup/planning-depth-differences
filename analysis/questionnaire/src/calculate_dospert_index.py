from argparse import ArgumentParser
from pathlib import Path

from costometer.utils import AnalysisObject

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp",
        default="QuestMain",
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

    # dospert
    # dospert-eb
    # dospert-rp
    # max(risk perception - risk benefit, 5 - risk benefit)
    analysis_obj
