"""Used for generating ground truth files for test environments"""
from pathlib import Path

from cluster_utils import create_test_env
from mouselab.env_utils import generate_ground_truth_file
from mouselab.mouselab import MouselabEnv

if __name__ == "__main__":
    for experiment_setting in ["cogsci_learning"]:
        create_test_env(experiment_setting)
        save_path = Path(__file__).parents[2].joinpath("data/inputs/exp_inputs/rewards")

        env = MouselabEnv.new_symmetric_registered(
            experiment_setting=experiment_setting
        )

        generate_ground_truth_file(
            env,
            num_ground_truths=100,
            save_path=save_path,
            file_name=experiment_setting,
            seed=91,
        )
