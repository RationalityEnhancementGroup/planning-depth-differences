import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
def calculate_vif(exogenous_df):
    exogenous_df = add_constant(exogenous_df)
    exogenous_df = exogenous_df.dropna()

    if "gender" in exogenous_df:
        exogenous_df = pd.concat(
            [exogenous_df, pd.get_dummies(exogenous_df["gender"])], axis=1
        )
        del exogenous_df["gender"]
        del exogenous_df["male"]

    return {col : variance_inflation_factor(exogenous_df.values, col_idx)
                for col_idx, col in enumerate(exogenous_df.columns)
            }