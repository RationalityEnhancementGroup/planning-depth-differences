"""
Runs Bayesian Model Comparison from SPM (in spm_BMS.m).

References (from spm_BMS.m):
% Stephan KE, Penny WD, Daunizeau J, Moran RJ, Friston KJ (2009)
% Bayesian Model Selection for Group Studies. NeuroImage 46:1004-1017
%
% Rigoux, L, Stephan, KE, Friston, KJ and Daunizeau, J. (2014)
% Bayesian model selection for group studies—Revisited.
% NeuroImage 84:971-85. doi: 10.1016/j.neuroimage.2013.08.065
"""
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from costometer.utils import get_static_palette
from costometer.utils.scripting_utils import standard_parse_args

NO_MATLAB = False
try:
    import matlab.engine
except ModuleNotFoundError:
    NO_MATLAB = True


def run_bms(optimization_data: pd.DataFrame, path_to_spm: Path = None) -> pd.DataFrame:
    """
    Run BMS on pandas dataframe.

    :param optimization_data: a dataframe containing the MLE or MAP parameter values
    for each participant (summed over trials) for all candidate models
    :param path_to_spm: path to SPM toolbox
    :return: dataframe including BMS outputs for each candidate model
    """
    # pivot dataframe
    pivoted_df = optimization_data.pivot(
        index="trace_pid", columns="Model Name", values="bic"
    )

    evidences = pivoted_df.values

    evidences = matlab.double(list([list(-0.5 * evidence) for evidence in evidences]))

    eng = matlab.engine.start_matlab()
    eng.addpath(str(path_to_spm.joinpath("spm12")))
    alpha, exp_r, xp = eng.spm_BMS(evidences, nargout=3)

    bms_df = pd.DataFrame(
        {
            "Model": list(pivoted_df),
            "Expected number of participants best explained by the model": [
                entry - 1 for entry in alpha._data
            ],
            "Model probability": exp_r._data,
            "Exceedance Probabilities": xp._data,
        }
    )
    return bms_df


def plot_bms_exceedance_probs(
    bms_out_df: pd.DataFrame,
    subdirectory,
    experiment_name,
    palette: Dict[str, Any] = None,
) -> None:
    """
    Plot BMS exceedance probabilities.

    :param bms_out_df: BMS results dataframe, including "Model" and \
    "Exceedance Probabilities"
    :param palette: palette as a dictionary models -> color
    :return: None
    """
    if palette is None:
        palette = get_static_palette(subdirectory, experiment_name)
    bar_order = (
        bms_out_df["Expected number of participants best explained by the model"]
        .sort_values()
        .index
    )
    plt.figure(figsize=(12, 8), dpi=80)
    sns.barplot(
        x="Exceedance Probabilities",
        y="Model",
        data=bms_out_df,
        palette=palette,
        order=bms_out_df.loc[bar_order, "Model"],
    )
    plt.title("Exceedance Probabilities")
    plt.xlabel("")


if __name__ == "__main__":
    """
    Example usage:
    python src/plot_bms.py -e MainExperiment
    """
    irl_path = Path(__file__).resolve().parents[4]
    analysis_obj, inputs, subdirectory = standard_parse_args(
        description=sys.modules[__name__].__doc__,
        irl_path=irl_path,
        filename=Path(__file__).stem,
    )

    optimization_data = analysis_obj.query_optimization_data()
    optimization_data = optimization_data[
        optimization_data.apply(
            lambda row: set(analysis_obj.analysis_details.excluded_parameters).issubset(
                row["model"]
            )
            or (row["Model Name"] == "Null"),
            axis=1,
        )
    ]

    irl_path.joinpath("data/bms/inputs/").mkdir(parents=True, exist_ok=True)
    irl_path.joinpath("data/bms/outputs/").mkdir(parents=True, exist_ok=True)
    if NO_MATLAB:
        if not irl_path.joinpath(
            f"data/bms/inputs/{inputs.experiment_name}.csv"
        ).is_file():
            pivoted_df = optimization_data.pivot(
                index="trace_pid", columns="Model Name", values="bic"
            )
            pivoted_df = pivoted_df.apply(lambda evidence: -0.5 * evidence)
            pivoted_df.to_csv(
                irl_path.joinpath(f"data/bms/inputs/{inputs.experiment_name}.csv")
            )
            quit()
        else:
            pivoted_df = pd.read_csv(
                irl_path.joinpath(f"data/bms/inputs/{inputs.experiment_name}.csv")
            )
            bms_df = pd.read_csv(
                irl_path.joinpath(f"data/bms/outputs/{inputs.experiment_name}.csv"),
                header=None,
            )

            import numpy as np

            for row_idx, _row in bms_df.iterrows():
                participant_row = bms_df.loc[row_idx]
                logging.info(
                    np.max(participant_row),
                    pivoted_df.columns[np.argmax(participant_row) + 1],
                )

    else:
        bms_df = run_bms(optimization_data, path_to_spm=irl_path)

    plot_bms_exceedance_probs(
        bms_df, subdirectory, experiment_name=inputs.experiment_name
    )
    plt.savefig(
        subdirectory.joinpath(
            f"figs/{inputs.experiment_name}_bms_exceedance_probabilities.png"
        ),
        bbox_inches="tight",
    )

    plot_bms_exceedance_probs(
        bms_df, subdirectory, experiment_name=inputs.experiment_name
    )
    plt.title("Exceedance Probabilities")
    plt.savefig(
        subdirectory.joinpath(
            f"figs/{inputs.experiment_name}_bms_exceedance_probabilities.png"
        ),
        bbox_inches="tight",
    )

    # prints info for latex table
    for field in [
        "Expected number of participants best explained by the model",
        "Model probability",
        "Exceedance Probabilities",
    ]:
        bms_df[field] = bms_df[field].apply(lambda entry: f"{entry:.2f}")
    logging.info(f"{' & '.join(bms_df)} \\\ \hline")  # noqa : W605

    # need as numeric rather than object for sorting
    bms_df["Expected number of participants best explained by the model"] = bms_df[
        "Expected number of participants best explained by the model"
    ].astype(float)

    for _row_idx, row in bms_df.sort_values(
        by="Expected number of participants best explained by the model",
        ascending=False,
    ).iterrows():
        logging.info(
            f"{' & '.join([val if not isinstance(val, float) else f'{val:0.2f}' for val in row.values])} \\\\"  # noqa : E501
        )
