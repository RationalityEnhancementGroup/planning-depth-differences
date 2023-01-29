"""
Tiny script to plot what a discrete prior looks like
"""
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from costometer.utils import get_temp_prior, set_font_sizes
from scipy import stats  # noqa needed for eval of rv string

if __name__ == "__main__":
    """
    Example usage:
    python src/plot_prior.py -p expon
    """
    parser = ArgumentParser()
    parser.add_argument(
        "-p",
        "--prior",
        dest="prior_file",
        default="uniform",
        help="Prior",
    )
    inputs = parser.parse_args()

    irl_path = Path(__file__).resolve().parents[4]
    subdirectory = irl_path.joinpath(f"analysis/methods/static")

    yaml_file = str(
        irl_path.joinpath(f"data/inputs/yamls/temperatures/{inputs.prior_file}.yaml")
    )
    yaml_path = Path(yaml_file).resolve()
    with open(str(yaml_path), "r") as stream:
        prior_inputs = yaml.safe_load(stream)

    temp_prior = get_temp_prior(
        rv=eval(prior_inputs["rv"]),
        possible_vals=prior_inputs["possible_temps"],
        inverse=prior_inputs["inverse"],
    )
    temp_prior_dict = dict(zip(temp_prior.vals, temp_prior.probs))

    set_font_sizes()
    # adapted from:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_discrete.html
    fig, ax = plt.subplots(1, 1)
    ax.plot(
        prior_inputs["possible_temps"],
        [temp_prior_dict[t] for t in prior_inputs["possible_temps"]],
        "ro",
        ms=12,
        mec="r",
    )
    ax.vlines(
        prior_inputs["possible_temps"],
        0,
        [temp_prior_dict[t] for t in prior_inputs["possible_temps"]],
        colors="r",
        lw=4,
    )
    ax.set_xscale("log")
    plt.xlabel("Possible temperatures (log scale)")
    plt.ylabel("Prior probability")
    print(subdirectory.joinpath(f"figs/prior_temp_{inputs.prior_file}.png"))
    plt.savefig(
        subdirectory.joinpath(f"figs/prior_temp_{inputs.prior_file}.png"),
        bbox_inches="tight",
    )
