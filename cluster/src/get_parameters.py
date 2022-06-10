"""
This is just a small helper file which helps with generating parameters
"""
import itertools
import sys
from pathlib import Path
from typing import Callable, List, Union

from costometer.utils import (
    create_parameter_grid,
    create_random_parameter_grid,
    save_combination_file,
)


def get_cluster_parameters(
    start: Union[float, int],
    stop: Union[float, int],
    step: Union[float, int],
    num_params: int,
    num_combinations: int = 10,
    random_seed: Union[int, Callable] = None,
) -> None:
    """
    This function creates two files:

    1. one with parameters on a grid
    2. one with parameters drawn randomly

    See documentation for each function in costometer.utils.cluster

    :param start: First value for grid, or start of range for random parameters
    :param stop: Last value for grid, or end of range for random parameters
    :param step: Space between points in grid parameters
    :param num_params: Number of parameters to output (# of parameters in cost function)
    :param random_seed: Random seed (if desired)
    :return: nothing, saves two parameter files in ../parameters/
    """

    location = Path(__file__).parents[1].joinpath("parameters/cost")
    # make directory if it doesn't already exist
    location.mkdir(parents=True, exist_ok=True)

    grid_params = create_parameter_grid(start, stop, step, num_params)
    save_combination_file(grid_params, "grid_params", location)

    random_params = create_random_parameter_grid(
        start, stop, num_params, seed=random_seed, num_combinations=num_combinations
    )
    save_combination_file(random_params, "non_grid_params", location)

    return None


def create_reduced_grid(reward: List[Union[int, float]], num_params: int) -> None:
    """
    Creates a reduced parameter grid file

    :param reward: rewards to combine to make this grid
    :param num_params: number of variables
    :return: None
    """
    location = Path(__file__).parents[1].joinpath("parameters/cost")
    # make directory if it doesn't already exist
    location.mkdir(parents=True, exist_ok=True)

    params = list(itertools.product(*[reward] * num_params))
    save_combination_file(params, "reduced_params", location)
    return None


if __name__ == "__main__":
    """
    Example call: python src/get_parameters.py 0 10.5 .5 2 91
    """
    start = float(sys.argv[1])
    stop = float(sys.argv[2])
    step = float(sys.argv[3])
    num_params = int(sys.argv[4])
    if len(sys.argv) == 6:
        random_seed = int(sys.argv[5])
    else:
        random_seed = None

    get_cluster_parameters(start, stop, step, num_params, random_seed=random_seed)
    create_reduced_grid([0.0, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0], num_params)
