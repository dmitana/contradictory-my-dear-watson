from typing import Union

from pandas import DataFrame, Series
from pandas.core.generic import NDFrame


def get_outliers(
    df: DataFrame,
    column: str
) -> Union[Series, DataFrame, NDFrame, None]:
    """
    Get outliers of given `column` in given `df` using Tukey test.

    :param df: dataframe.
    :param column: dataframe's column to get outliers of.
    :return: dataframe containing outliers.
    """
    iqr = df[column].quantile(q=0.75) - df[column].quantile(q=0.25)
    upper_bound = df[column].quantile(q=0.75) + 1.5 * iqr
    lower_bound = df[column].quantile(q=0.25) - 1.5 * iqr

    print(f'lower bound: {lower_bound}\nupper bound: {upper_bound}')

    return df[(df[column] > upper_bound) | (df[column] < lower_bound)]
