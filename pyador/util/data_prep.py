import pandas as pd
import numpy as np


def integrity_check(df):
    if isinstance(df, pd.DataFrame):
        return True
    else:
        raise TypeError("Pyador only supports pandas dataframe")


'''
check the missing percentage
impute the missing values 
'''


def miss_check(df, imputation):
    '''

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
            print("{col} missing percentage is {miss}%".format(col=col,
                                                               miss=mis_perc))
            if imputation == "mean":
                df[col].fillna(df[col].mean, inplace=True)
            else:
                df[col].fillna(0, inplace=True)


def cat_to_num(df):
    pass
