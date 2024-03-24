import sys
from pathlib import Path

from costometer.utils import standard_parse_args

if __name__ == "__main__":
    irl_path = Path(__file__).resolve().parents[4]
    analysis_obj, inputs, subdirectory = standard_parse_args(
        description=sys.modules[__name__].__doc__,
        irl_path=irl_path,
        filename=Path(__file__).stem,
    )

    trial_by_trial_df = analysis_obj.get_trial_by_trial_likelihoods()
