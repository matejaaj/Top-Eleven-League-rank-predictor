import pandas as pd
import numpy as np

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df, columns_to_normalize, columns_to_drop, groupby_column='league_id'):
    """
    Preprocess the given DataFrame by normalizing selected columns based on a grouping column
    and dropping specified unnecessary columns.
    
    Parameters:
    - df: pandas.DataFrame, the input dataframe to preprocess.
    - columns_to_normalize: list of str, columns to normalize.
    - columns_to_drop: list of str, columns to drop from the dataframe.

    Returns:
    - df: pandas.DataFrame, the preprocessed dataframe.
    """
    
    for col in columns_to_normalize:
        normalized_col_name = f'normalized_{col}'
        
        if col in df.columns:
            normalized_values = df.groupby(groupby_column)[col].transform(lambda x: (x - x.mean()) / x.std())
            df[normalized_col_name] = normalized_values.fillna(0)

    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=columns_to_drop)
    
    return df


