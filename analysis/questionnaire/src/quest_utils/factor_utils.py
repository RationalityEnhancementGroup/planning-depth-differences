import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

scale = StandardScaler(with_mean=True, with_std=True)

col_dict = {
    "alcohol": "AUDIT",
    "anxiety": "STAI",
    "apathy": "AES",
    "bis": "BIS",
    "eat": "EAT",
    "leb": "LSAS",
    "ocir": "OCI",
    "scz": "SCZ",
    "schizo": "SCZ",
    "zung": "SDS",
}


def columns_to_gillan(col, col_dict):
    col_parts = col.split(".")
    if col_parts[0] in col_dict:
        col_parts[0] = col_dict[col_parts[0]]
        return "_".join(col_parts)
    else:
        return col


def load_weights(weights_path, wise_weights=False):
    if wise_weights:
        weights = pd.read_csv(weights_path, index_col="item")
        if "AD" not in list(weights):
            weights = weights.rename(
                {
                    "Anxiety/depression": "AD",
                    "Compulsivity": "CIT",
                    "Social withdrawal": "SW",
                },
                axis="columns",
            )
            weights = weights.rename(
                {
                    "Anxiety/depression": "AD",
                    "Compulsivity": "CIT",
                    "Social withdrawal": "SW",
                },
                axis="columns",
            )
        weights.index = weights.index.map(
            lambda index: columns_to_gillan(index, col_dict)
        )
        weights = weights[weights.sum(axis=1) > 0]
    else:
        weights = pd.read_csv(weights_path, index_col=0)
    return weights


def get_psychiatric_scores(individual_items, weights, scale_cols=True):
    individual_items.columns = [
        columns_to_gillan(col, col_dict) for col in individual_items.columns
    ]
    # combine both lsas measures for scoring if gillan
    if "LSAS_25" in individual_items:
        for col in range(25, 49):
            individual_items["LSAS_{}".format(col - 24)] = (
                individual_items["LSAS_{}".format(col - 24)]
                + individual_items["LSAS_{}".format(col)]
            )
            del individual_items["LSAS_{}".format(col)]
    else:
        print("LSAS already combined")

    # standard scale columns
    if scale_cols:
        individual_items = pd.DataFrame(
            scale.fit_transform(individual_items),
            columns=individual_items.columns,
            index=individual_items.index,
        )

    for col in list(weights):
        individual_items[col] = individual_items.apply(
            lambda row: np.sum(
                [row[item] * weights.loc[item, col] for item in weights.index]
            ),
            axis=1,
        )

    # TODO is there a better way to do this
    return individual_items[list(weights)]
