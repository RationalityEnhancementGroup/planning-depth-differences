import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from mouselab.graph_utils import annotate_mdp_graph, get_structure_properties
from mouselab.mouselab import MouselabEnv

sys.path.append(str(Path(__file__).parents[1].joinpath("cluster/src")))
from cluster_utils import create_test_env  # noqa : E402

if __name__ == "__main__":
    for experiment_setting in [
        "small_test_case",
        "reduced_leaf",
        "reduced_middle",
        "reduced_root",
        "reduced_variance",
    ]:
        create_test_env(experiment_setting)

        env = MouselabEnv.new_symmetric_registered(
            experiment_setting=experiment_setting
        )

        # first we need to load experiment setting file and then structure file
        with open(
            Path(__file__)
            .parents[1]
            .joinpath(
                f"data/inputs/yamls/experiment_settings/{experiment_setting}.yaml"
            ),
            "r",
        ) as stream:
            args = yaml.safe_load(stream)

        if "structure" in args:
            with open(
                Path(__file__)
                .parents[1]
                .joinpath(f"data/inputs/exp_inputs/structure/{args['structure']}.json"),
                "rb",
            ) as f:
                structure_data = json.load(f)

            structure_dicts = get_structure_properties(structure_data)
        else:
            structure_dicts = None

        env.mdp_graph = annotate_mdp_graph(env.mdp_graph, structure_dicts)

        for action in list(env.actions(env._state)):
            if action != env.term_action:
                env.step(action)

        plt.figure()
        env._render(use_networkx=True)
        plt.show()
