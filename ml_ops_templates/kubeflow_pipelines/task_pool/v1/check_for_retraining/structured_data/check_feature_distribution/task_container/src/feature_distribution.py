"""

Check feature distributions of structured (tabular) data for significant changes to use it for continuous training approach

"""

import numpy as np
import pandas as pd

from custom_logger import Log
from scipy.stats import chi2, chi2_contingency, mannwhitneyu, kstest
from typing import List


class FeatureDistributionException(Exception):
    """
    Class for handling exceptions for class FeatureDistribution
    """
    pass


class FeatureDistribution:
    """
    Class for checking feature distributions of structured (tabular) data
    """
    def __init__(self,
                 previous_observation_values: np.array,
                 previous_observation_name: str,
                 current_observation_values: np.array,
                 current_observation_name: str,
                 bootstrapping_n_samples: int = 10,
                 bootstrapping_replace: bool = False,
                 p: float = 0.95
                 ):
        """
        :param previous_observation_values: np.array
            Values of the previously observed feature

        :param previous_observation_name: str
            Name of the previously observed feature

        :param current_observation_values: np.array
            Values of the currently observed feature

        :param current_observation_name: str
            Name of the currently observed feature

        :param bootstrapping_n_samples: int
            Number of samples to draw in bootstrapping

        :param bootstrapping_replace: bool
            Whether to allow of disallow sampling of the same row more than once in bootstrapping

        :param p: float
            Probability value for rejecting hypothesis
        """
        self.df: pd.DataFrame = pd.DataFrame()
        _n_cases_previous_obs: int = previous_observation_values.shape[0]
        _n_cases_current_obs: int = current_observation_values.shape[0]
        if _n_cases_previous_obs == _n_cases_current_obs:
            self.x: str = previous_observation_name
            self.y: List[str] = [current_observation_name]
            self.df[previous_observation_name] = previous_observation_values
            self.df[current_observation_name] = current_observation_values
        else:
            if _n_cases_previous_obs > _n_cases_current_obs:
                self.x: str = current_observation_name
                _feature_name: str = previous_observation_name
                _feature_values: np.array = previous_observation_values
                _sample_size: int = _n_cases_current_obs
                self.df[current_observation_name] = current_observation_values
            else:
                self.x: str = previous_observation_name
                _feature_name: str = current_observation_name
                _feature_values: np.array = current_observation_values
                _sample_size: int = _n_cases_previous_obs
                self.df[previous_observation_name] = previous_observation_values
            _df: pd.DataFrame = self._bootstrapping(feature_name=_feature_name,
                                                    feature_values=_feature_values,
                                                    sample_size=_sample_size,
                                                    n_samples=bootstrapping_n_samples,
                                                    replace=bootstrapping_replace
                                                    )
            self.y: List[str] = self.df.columns.tolist()
            self.df = pd.concat(objs=[self.df, _df], axis=1)
        self.p: float = 1 - p

    @staticmethod
    def _bootstrapping(feature_name: str,
                       feature_values: np.array,
                       sample_size: int,
                       n_samples: int = 10,
                       replace: bool = False
                       ) -> pd.DataFrame:
        """
        Generate equal size samples for testing independent distribution of different observations

        :param feature_name: str
            Name of the feature used for sampling

        :param feature_values: np.array
            Values of given feature

        :param sample_size: int
            Sample size to draw

        :param n_samples: int
            Number of samples to draw

        :param replace: bool
            Whether to allow of disallow sampling of the same row more than once

        :return: pd.DataFrame
            Sample data set
        """
        _df_x: pd.DataFrame = pd.DataFrame(data={feature_name: feature_values})
        _df_y: pd.DataFrame = pd.DataFrame()
        Log().log(msg=f'Bootstrapping: Feature={feature_name}, Cases={_df_x.shape[0]}, Sample size={sample_size}, Samples={n_samples}, Replacement={replace}')
        for i in range(0, n_samples, 1):
            _df_y[f'{feature_name}_{i}'] = _df_x[feature_name].sample(n=sample_size, replace=replace, random_state=1234).values
        return _df_y

    def _chi_squared_independence_test(self, y: str) -> dict:
        """
        Test the independency of two categorical features using Chi-Squared test

        :param y: str
            Name of the second feature

        :return dict
            Statistical test results (number of cases, test statistic, p-value, reject)
        """
        _cross_tab: pd.DataFrame = pd.crosstab(self.df[self.x],
                                               self.df[y],
                                               margins=True,
                                               margins_name="Total"
                                               )
        _chi_square: float = 0
        _rows: np.array = self.df[self.x].unique()
        _columns: np.array = self.df[y].unique()
        for i in _columns:
            for j in _rows:
                _o: float = _cross_tab[i][j]
                _e: float = _cross_tab[i]['Total'] * _cross_tab['Total'][j] / _cross_tab['Total']['Total']
                _chi_square += (_o - _e) ** 2 / _e
        _p_value: float = 1 - chi2.cdf(_chi_square, (len(_rows) - 1) * (len(_columns) - 1))
        if _p_value <= self.p:
            _reject: bool = True
        else:
            _reject: bool = False
        Log().log(msg=f'Chi-Squared: X={self.x}, Y={y}, Cases={self.df.shape[0]}, Statistic={_chi_square}, P-Value={_p_value}, Reject={_reject}')
        return {'cases': self.df.shape[0],
                'test_statistic': _chi_square,
                'p_value': _p_value,
                'reject': _reject
                }

    def _mann_whitney_u_test(self, y: str) -> dict:
        """
        Test the independency of two continuous features using Mann-Whitney-U test

        :param y: str
            Name of the second feature

        :return dict
            Statistical test results (number of cases, test statistic, p-value, reject)
        """
        _statistic, _p_value = mannwhitneyu(self.df[self.x].values, self.df[y].values)
        if _p_value <= self.p:
            _reject: bool = True
        else:
            _reject: bool = False
        Log().log(msg=f'Mann-Whitney-U: X={self.x}, Y={y}, Cases={self.df.shape[0]}, Statistic={_statistic}, P-Value={_p_value}, Reject={_reject}')
        return {'cases': self.df.shape[0],
                'test_statistic': _statistic,
                'p_value': _p_value,
                'reject': _reject
                }

    def main(self, meth: str) -> bool:
        """
        Check feature distribution

        :param meth: str
            Name of the statistical test method
                -> chi2_test: Chi-square independent test (for categorical features)
                -> mann_whitney_u_test: Mann-Whitney-U test (for continuous features)

        :return: bool
            Whether to reject independence or not
        """
        for y in self.y:
            if meth == 'chi2_test':
                _test_results: dict = self._chi_squared_independence_test(y=y)
            elif meth == 'mann_whitney_u_test':
                _test_results: dict = self._mann_whitney_u_test(y=y)
            else:
                raise FeatureDistributionException(f'Statistical testing method ({meth}) not supported')
            if _test_results.get('reject'):
                return True
        return False
