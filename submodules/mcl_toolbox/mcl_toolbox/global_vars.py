import os
import pickle
from pathlib import Path

file_location = Path(__file__).parents[0]


# RenameUnpickler from https://stackoverflow.com/a/53327348
class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        # we need these modules in order to load pickled files depending on them,
        # pickled with earlier versions of the code
        if module == "learning_utils":
            renamed_module = "mcl_toolbox.utils.learning_utils"
        elif module == "distributions":
            renamed_module = "mouselab.distributions"

        return super(RenameUnpickler, self).find_class(renamed_module, name)


def pickle_load(file_path):
    """
    Load the pickle file located at 'filepath'
    Params:
        file_path  -- Location of the file to be loaded.
    Returns:
        Unpickled object
    """
    if not os.path.exists(file_path):
        head, tail = os.path.split(__file__)
        if file_path[0] == "/":
            new_path = os.path.join(head, file_path[1:])
        else:
            new_path = os.path.join(head, file_path)
        if os.path.exists(new_path):
            file_path = new_path
        else:
            raise FileNotFoundError(f"{file_path} not found.")
    file_obj = open(file_path, "rb")
    return RenameUnpickler(file_obj).load()


class strategies:
    strategy_space = pickle_load(file_location / "data" / "strategy_space.pkl")
    strategy_weights = pickle_load(file_location / "data" / "microscope_weights.pkl")


class features:
    microscope = pickle_load(
        file_location / "data" / "microscope_features.pkl"
    )  # this is 51 features
