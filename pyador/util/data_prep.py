import pandas as pd
import numpy as np

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype


def numerical_check(df):
    ''' function to check if the dataframe are fully numerical

    :param df:
    :return:
    '''

    # check if it is a pandas framework or series
    dtype = integrity_check(df)

    # check if the series is numeric
    if dtype == 'series':
        return is_numeric_dtype(df)

    # select all numerical variables and get the length
    num_vars = df.select_dtypes(include=[np.number])
    l_num_vars = num_vars.shape[1]

    if l_num_vars == df.shape[1]:
        return True
    else:
        return False


def integrity_check(df):
    ''' check if the input data is compliant with the model requirement

    :param df:
    :return:
    '''
    if isinstance(df, pd.DataFrame):
        return 'df'
    elif isinstance(df, pd.Series):
        return 'series'
    else:
        raise TypeError("Pyador only supports pandas dataframe or series")


def miss_check(df, imputation, verbose):
    '''check the missing percentage
    impute the missing values if necessary

    :param df:
    :param imputation: "zero" or "mean"
    :return:
    '''
    n_df = df.shape[0]
    cols = df.columns.tolist()

    for col in cols:
        missing = n_df - np.count_nonzero(df[col].isnull().values)
        mis_perc = 100 - float(missing) / n_df * 100

        if mis_perc > 0:
            if verbose:
                print("{col} missing percentage is {miss}%" \
                      .format(col=col, miss=mis_perc))
            if imputation == "mean":
                df[col].fillna(df[col].mean, inplace=True)
            else:
                df[col].fillna(0, inplace=True)


def cat_to_num(df):
    pass
