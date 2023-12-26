"""

Data health check: missing value analysis, invariant features, duplicated features

"""

import numpy as np
import pandas as pd

from custom_logger import Log
from typing import List

INVALID_VALUES: list = ['nan', 'NaN', 'NaT', np.nan, 'none', 'None', 'inf', '-inf', np.inf, -np.inf]


class DataHealthCheckException(Exception):
    """
    Class for handling exception for class DataHealthCheck
    """
    pass


class DataHealthCheck:
    """
    Class for checking data health
    """
    def __init__(self, df: pd.DataFrame, feature_names: List[str]):
        """
        :param df: pd.DataFrame
            Data set

        :param feature_names: str
            Name of the features to analyse
        """
        self.df: pd.DataFrame = df
        self.feature_names: List[str] = feature_names
        self.n_cases: int = self.df.shape[0]
        self.duplicates_features: List[str] = self.df.iloc[:, np.where(self.df.transpose().duplicated().values)[0]].columns.tolist()

    def _convert_invalid_to_nan(self) -> None:
        """
        Convert several invalid values to missing values
        """
        for invalid in INVALID_VALUES:
            self.df.replace(to_replace=invalid, value=np.nan, inplace=True)
            Log().log(msg=f'Converted invalid values ({invalid}) to missing values (NaN)')

    def _is_duplicated(self, feature_name: str) -> bool:
        """
        Check whether a feature is duplicated or not

        :param feature_name: str
            Name of the feature

        :return: bool
            Whether given feature is duplicated or not
        """
        if feature_name in self.duplicates_features:
            _is_duplicated: bool = True
        else:
            _is_duplicated: bool = False
        Log().log(msg=f'Feature "{feature_name}" is duplicated')
        return _is_duplicated

    def _is_invariant(self, feature_name: str) -> bool:
        """
        Check whether a feature is invariant or not

        :param feature_name: str
            Name of the feature

        :return: bool
            Whether given feature is invariant or not
        """
        _n_unique_values: int = len(self.df[feature_name].unique().tolist())
        Log().log(msg=f'Feature "{feature_name}" has {_n_unique_values} unique values')
        return _n_unique_values == 1

    def _get_nan_idx(self, feature_name: str) -> List[int]:
        """
        Get index values of existing missing values (nan)

        :param feature_name: str
            Name of the feature

        :return: List[int]
            Index values of missing values
        """
        return np.where(pd.isnull(self.df[feature_name]))[0].tolist()

    def _missing_value_analysis(self, feature_name: str) -> dict:
        """
        Missing value analysis

        :param feature_name: str
            Name of the feature

        :return: dict
            Number of detected missing values and index values
        """
        _nan_idx: List[int] = self._get_nan_idx(feature_name=feature_name)
        _n_missing_values: int = len(_nan_idx)
        if _n_missing_values == 0:
            _prop_missing_values: float = 0.0
        else:
            _prop_missing_values: float = round(number=len(_nan_idx) / self.n_cases, ndigits=4)
        Log().log(msg=f'Feature {feature_name} has {_n_missing_values} ({_prop_missing_values * 100} %) missing values (NaN)')
        return dict(number_of_missing_values=_n_missing_values,
                    proportion_of_missing_values=_prop_missing_values,
                    nan_idx=_nan_idx
                    )

    def main(self) -> dict:
        """
        Run data health check

        :return: dict
            Results of missing value analysis, invariant analysis
        """
        self._convert_invalid_to_nan()
        _data_health_check: dict = {}
        for feature in self.feature_names:
            _data_health_check.update({feature: dict(missing_value_analysis=self._missing_value_analysis(feature_name=feature),
                                                     invariant=self._is_invariant(feature_name=feature),
                                                     duplicated=self._is_duplicated(feature_name=feature)
                                                     )
                                       })
        return _data_health_check
