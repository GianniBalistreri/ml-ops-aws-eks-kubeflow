"""

Imputation: impute missing values

"""

import copy
import numpy as np
import pandas as pd

from custom_logger import Log
from typing import Dict, List, Tuple, Union


class ImputationException(Exception):
    """
    Class for handling exception for class Imputation
    """
    pass


class Imputation:
    """
    Class for imputing missing values
    """
    def __init__(self, df: pd.DataFrame, **kwargs):
        """
        :param df: pd.DataFrame
            Data set

        :param kwargs: dict
            Key-word arguments
        """
        self.imp_config: dict = {}
        self.df: pd.DataFrame = df
        self.missing_value_count: int = 0
        self.missing_value_count_by_feature: Dict[str, int] = {}
        self.kwargs: dict = {} if kwargs is None else kwargs
        self._check_missing_values()

    def _check_missing_values(self) -> None:
        """
        Check which feature contains missing values

        :return:
        """
        _missing_value_count_df: pd.Series = pd.isnull(self.df[self.df.columns.tolist()]).astype(int).sum()
        self.missing_value_count = _missing_value_count_df.sum()
        self.missing_value_count_by_feature = _missing_value_count_df.to_dict()
        Log().log(msg=f'Total number of missing values in data set: {self.missing_value_count}')

    def _mice(self) -> None:
        """
        Multiple imputation by chained equation for impute missing at random (MAR) patterns using predictions of machine learning models
        """
        raise ImputationException('MICE algorithm not supported')

    def _random(self, feature: str, m: int = 3, convergence_threshold: float = 0.99) -> np.ndarray:
        """
        Multiple imputation using randomly generated values within range of observations

        :param feature: str
            Name of the feature to impute

        :param m: int
            Number of chains (multiple imputation)

        :param convergence_threshold: float
            Convergence threshold used for multiple imputation

        :return: np.ndarray
            Imputed values for given feature
        """
        _threshold: float = convergence_threshold if (convergence_threshold > 0) and (
                    convergence_threshold < 1) else 0.99
        _unique_values: np.ndarray = self.df[feature][~self.df[feature].isnull()].unique()
        _mis_idx: np.ndarray = np.where(pd.isnull(self.df[feature]))[0]
        if str(_unique_values.dtype).find('int') < 0 and str(_unique_values.dtype).find('float') < 0:
            _unique_values = _unique_values.astype(dtype=float)
        _value_range: Tuple[float, float] = (min(_unique_values), max(_unique_values))
        _std: float = self.df[feature].std()
        _threshold_range: Tuple[float, float] = (_std - (_std * (1 - _threshold)), _std + (_std * (1 - _threshold)))
        _m: List[List[float]] = []
        _std_theta: list = []

        for n in range(m):
            _data: np.ndarray = self.df[feature].values.copy()
            _imp_value: list = []
            for idx in _mis_idx:
                if any(_unique_values % 1 != 0):
                    _imp_value.append(np.random.uniform(low=_value_range[0], high=_value_range[1]))
                else:
                    _imp_value.append(np.random.randint(low=int(_value_range[0]), high=int(_value_range[1]) + 1))
                _data[idx] = _imp_value[-1]
            _std_theta.append(abs(_std - np.std(_data)))
            if _std_theta[-1] >= _threshold_range[0] and _std_theta[-1] <= _threshold_range[1]:
                break
            _m.append(_imp_value)
        _best_imputation: list = _m[_std_theta.index(min(_std_theta))]
        _imp_data: pd.DataFrame = pd.DataFrame()
        _imp_data[feature] = self.df[feature].values
        for i, idx in enumerate(_mis_idx):
            _imp_data.loc[idx, feature] = _best_imputation[i]
            self.imp_config[feature]['imp_value'].append(_best_imputation[i])

        _std_diff: float = 1 - round(_std / self.df[feature].std(), 2)
        _msg_element: str = 'in' if _std_diff > 0 else 'de'
        Log().log(msg=f'Variance of feature ({feature}) {_msg_element}creases by {_std_diff * 100}%')
        return _imp_data[feature].values

    def main(self,
             feature_names: List[str],
             imp_meth: str,
             multiple_meth: str = 'random',
             single_meth: str = 'constant',
             constant_value: Union[int, float] = None,
             m: int = 3,
             convergence_threshold: float = 0.99,
             mice_config: dict = None
             ) -> pd.DataFrame:
        """
        Run missing value imputation

        :param feature_names: str
            Name of the features to impute

        :param imp_meth: str
            Name of the imputation method
                -> single: Single imputation
                -> multiple: Multiple Imputation

        :param multiple_meth: str
            Name of the multiple imputation method
                -> mice: Multiple Imputation by Chained Equation
                -> random: Random

        :param single_meth: str
            Name of the single imputation method
                -> constant: Constant value
                -> min: Minimum observed value
                -> max: Maximum observed value
                -> median: Median of observed values
                -> mean: Mean of observed values
                -> interpolation: Linear interpolation between values

        :param constant_value: Union[int, float]
            Constant imputation value used for single imputation method constant

        :param m: int
            Number of chains (multiple imputation)

        :param convergence_threshold: float
            Convergence threshold used for multiple imputation

        :param mice_config: dict
            Multiple imputation by chained equation method as well as predictor mapping for each feature

        :return: pd.DataFrame
            Imputed data set
        """
        _imp_df: pd.DataFrame = pd.DataFrame()
        for feature in feature_names:
            if self.missing_value_count_by_feature.get(feature) == 0:
                _imp_df[feature] = self.df[feature]
                Log().log(msg=f'No missing values of feature {feature} detected')
            else:
                self.imp_config.update({feature: dict(imp_meth_type=imp_meth)})
                if imp_meth == 'single':
                    self.imp_config[feature].update({'imp_meth': single_meth})
                    if single_meth == 'constant':
                        self.imp_config[feature].update({'imp_value': [constant_value]})
                        _imp_df[feature] = self.df[feature].fillna(value=constant_value, inplace=False)
                    elif single_meth == 'min':
                        _imp_value: float = self.df[feature].min(skipna=True)
                        self.imp_config[feature].update({'imp_value': [_imp_value]})
                        _imp_df[feature] = self.df[feature].fillna(value=_imp_value, inplace=False)
                    elif single_meth == 'max':
                        _imp_value: float = self.df[feature].max(skipna=True)
                        self.imp_config[feature].update({'imp_value': [_imp_value]})
                        _imp_df[feature] = self.df[feature].fillna(value=_imp_value, inplace=False)
                    elif single_meth == 'median':
                        _imp_value: float = self.df[feature].median(skipna=True)
                        self.imp_config[feature].update({'imp_value': [_imp_value]})
                        _imp_df[feature] = self.df[feature].fillna(value=_imp_value, inplace=False)
                    elif single_meth == 'mean':
                        _imp_value: float = self.df[feature].mean(skipna=True)
                        self.imp_config[feature].update({'imp_value': [_imp_value]})
                        _imp_df[feature] = self.df[feature].fillna(value=_imp_value, inplace=False)
                    elif single_meth == 'interpolation':
                        _imp_df[feature] = self.df[feature].interpolate(method='linear' if self.kwargs.get('method') is None else self.kwargs.get('method'),
                                                                        axis=1,
                                                                        limit=self.kwargs.get('limit'),
                                                                        inplace=False,
                                                                        limit_direction='forward' if self.kwargs.get('limit_direction') is None else self.kwargs.get('limit_direction'),
                                                                        limit_area='inside' if self.kwargs.get('limit_area') is None else self.kwargs.get('limit_area')
                                                                        )
                    elif single_meth in ['ffill', 'bfill']:
                        _imp_df[feature] = self.df[feature].fillna(method=single_meth, inplace=False)
                    else:
                        raise ImputationException(f'Single imputation method ({single_meth}) not supported')
                    _std_diff: float = 1 - round(_imp_df[feature].std() / self.df[feature].std())
                    _msg_element: str = 'in' if _std_diff >= 0 else 'de'
                    Log().log(msg=f'Variance of feature ({feature}) {_msg_element}creases by {_std_diff}%')
                    Log().log(msg=f'Missing values of feature {feature} imputed by applying single imputation method ({single_meth})')
                elif imp_meth == 'multiple':
                    self.imp_config[feature].update({'imp_meth': multiple_meth, 'imp_value': []})
                    if multiple_meth == 'mice':
                        raise ImputationException(f'Method ({multiple_meth}) not supported')
                    elif multiple_meth == 'random':
                        _imp_df[feature] = self._random(feature=feature, m=m, convergence_threshold=convergence_threshold)
                    else:
                        raise ImputationException(f'Multiple imputation method ({multiple_meth}) not supported')
                    Log().log(msg=f'Missing values of feature {feature} imputed by applying multiple imputation method ({multiple_meth})')
                else:
                    raise ImputationException(f'Imputation method ({imp_meth}) not supported')
        return _imp_df
