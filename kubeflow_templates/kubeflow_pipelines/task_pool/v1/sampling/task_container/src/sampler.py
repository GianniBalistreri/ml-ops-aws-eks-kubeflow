"""

Data sampling

"""

import pandas as pd
import random

from custom_logger import Log
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from typing import Dict, List, Tuple, Union


class MLSamplerException(Exception):
    """
    Class for handling exceptions for class MLSampler
    """
    pass


class MLSampler:
    """
    Class for data sampling specialized for supervised machine learning
    """
    def __init__(self,
                 df: pd.DataFrame,
                 target: str,
                 features: List[str] = None,
                 time_series_feature: str = None,
                 train_size: float = 0.8,
                 validation_size: float = 0.1,
                 random_sample: bool = True,
                 stratification: bool = False,
                 seed: int = 1234
                 ):
        """
        :param df: Pandas DataFrame
            Data set for sampling

        :param target: str
            Name of the target feature

        :param features: List[str]
            Name of the features

        :param time_series_feature: str
            Name of the time series feature to sort by

        :param train_size: float
            Size of the training set

        :param validation_size: float
            Size of the validation set

        :param random_sample: bool
            Whether to sample randomly or not

        :param stratification: bool
            Whether to stratify data set or not

        :param seed: int
            Seed value
        """
        self.seed: int = seed if seed >= 0 else 1234
        self.df: pd.DataFrame = df
        if target not in self.df.columns:
            raise MLSamplerException(f'Target feature ({target}) not found in data set')
        self.target = target
        self.features: List[str] = list(self.df.columns) if (features is None or len(features) == 0) else features
        if self.target in self.features:
            del self.features[self.features.index(self.target)]
        for ft in self.features:
            if ft not in self.df.columns:
                raise MLSamplerException(f'Feature ({ft}) not found in data set')
        self.time_series_feature: str = time_series_feature
        self.train_size: float = train_size
        self.test_size: float = 1 - self.train_size
        self.validation_size: float = validation_size
        self.random_sample: bool = random_sample
        self.stratification: bool = stratification

    def _get_freq_of_target_class_value(self,
                                        df: pd.DataFrame,
                                        target_class_value: Union[str, int]
                                        ) -> Tuple[int, float]:
        """
        Get absolute and relative frequency values of given target class value

        :param df:
        :param target_class_value:
        :return:
        """
        _abs_freq: pd.Series = df[self.target].value_counts(normalize=False)
        _rel_freq: pd.Series = df[self.target].value_counts(normalize=True)
        _labels: list = _rel_freq.index.values.tolist()
        if target_class_value not in _labels:
            raise MLSamplerException(f'Given target value ({target_class_value}) not found')
        _count_values: List[float] = _abs_freq.values.tolist()
        _proportion_values: List[float] = _rel_freq.values.tolist()
        _current_count: int = _abs_freq[_labels.index(target_class_value)]
        _current_proportion: float = _proportion_values[_labels.index(target_class_value)]
        return _current_count, _current_proportion

    def down_sampling(self, target_class_value: Union[str, int], target_proportion: float) -> pd.DataFrame:
        """
        Down sample specific ranges of target values

        :return:
        """
        if 0 < target_proportion < 1:
            _current_count, _current_proportion = self._get_freq_of_target_class_value(df=self.df,
                                                                                       target_class_value=target_class_value
                                                                                       )
            Log().log(msg=f'Target feature: {self.target} -> target class value: {target_class_value} -> absolute frequency = {_current_count}, relative frequency = {_current_proportion}')
            if _current_proportion < target_proportion:
                raise MLSamplerException(f'Given target proportion ({target_proportion}) is higher than current proportion ({_current_proportion})')
            _target_absolute_count: int = _current_count - (self.df.shape[0] - round((target_proportion / _current_proportion) * self.df.shape[0]))
            _sample_idx: list = random.sample(population=list(self.df.loc[self.df[self.target] == target_class_value, :]), k=_target_absolute_count)
            _df_other_target_classes: pd.DataFrame = self.df.loc[self.df[self.target] != target_class_value, :]
            _df: pd.DataFrame = pd.concat(objs=[_df_other_target_classes, self.df.iloc[_sample_idx, :]], axis=0)
            _new_current_count, _new_current_proportion = self._get_freq_of_target_class_value(df=_df,
                                                                                               target_class_value=target_class_value
                                                                                               )
            Log().log(msg=f'Down-sample target feature: {self.target} -> target class value: {target_class_value} -> new absolute frequency = {_new_current_count}, new relative frequency = {_new_current_proportion}, requested relative frequency = {target_proportion}')
            return _df
        else:
            raise MLSamplerException(f'Given target proportion ({target_proportion}) is invalid')

    def k_fold_cross_validation(self, k: int = 5) -> dict:
        """
        K-Fold Cross-validation

        :param k: int
            Number of k-folds

        :return dict:
            Cross-validated train and test split for both target and predictors
        """
        _kfold_sample: dict = {}
        _k: int = k if k > 0 else 5
        if self.stratification:
            _kfold: StratifiedKFold = StratifiedKFold(n_splits=_k, shuffle=self.random_sample, random_state=self.seed)
        else:
            _kfold: KFold = KFold(n_splits=_k, shuffle=self.random_sample, random_state=self.seed)
        _counter: int = 0
        for train, test in _kfold.split(X=self.df.loc[:, self.features], y=self.df.loc[:, self.target], groups=None):
            _kfold_sample.update({f'x_train_k{_counter}': train, f'x_test_k{_counter}': test})
            Log().log(msg=f'K-fold cross-validation sampling: K={_counter}, Train cases={train.shape[0]}, Test cases={test.shape[0]}')
            _counter += 1
        return _kfold_sample

    def train_test_sampling(self) -> dict:
        """
        Data sampling into train & test data

        :return dict:
            Train and test split for both target and predictors
        """
        #if self.stratification:
        #    _stratification: np.array = self.df[self.target].values
        #else:
        #    _stratification = None
        _x_train, _x_test, _y_train, _y_test = train_test_split(self.df[self.features],
                                                                self.df[self.target],
                                                                test_size=self.test_size,
                                                                train_size=self.train_size,
                                                                random_state=self.seed,
                                                                shuffle=self.random_sample,
                                                                #stratify=_stratification
                                                                )
        if self.validation_size > 0:
            _x_train_, _x_val, _y_train_, _y_val = train_test_split(_x_train,
                                                                    _y_train,
                                                                    test_size=self.validation_size,
                                                                    train_size=1-self.validation_size,
                                                                    random_state=self.seed,
                                                                    shuffle=self.random_sample
                                                                    )
            Log().log(msg=f'Train-test sampling: Validation cases={_x_val.shape[0]}')
        else:
            _x_train_ = _x_train
            del _x_train
            _x_val = None
            _y_train_ = _y_train
            del _y_train
            _y_val = None
        Log().log(msg=f'Train-test sampling: Train cases={_x_train_.shape[0]}, Test cases={_x_test.shape[0]}')
        return dict(x_train=_x_train_,
                    x_test=_x_test,
                    y_train=_y_train_,
                    y_test=_y_test,
                    x_val=_x_val,
                    y_val=_y_val
                    )

    def time_series_sampling(self) -> dict:
        """
        Timeseries data sampling into train & test data

        :return: dict:
            Train and test split for both target and predictors
        """
        if self.time_series_feature is None or self.time_series_feature not in self.df.columns:
            raise MLSamplerException(f'Time series feature ({self.time_series_feature}) not found in data set')
        self.df.sort_values(by=self.time_series_feature, axis=1, ascending=True, inplace=True)
        self.random_sample = False
        return self.train_test_sampling()

    def up_sampling(self, target_class_value: Union[str, int], target_proportion: float) -> pd.DataFrame:
        """
        Up sample specific ranges of target values

        :return:
        """
        if 0 < target_proportion < 1:
            _current_count, _current_proportion = self._get_freq_of_target_class_value(df=self.df,
                                                                                       target_class_value=target_class_value
                                                                                       )
            Log().log(msg=f'Target feature: {self.target} -> target class value: {target_class_value} -> absolute frequency = {_current_count}, relative frequency = {_current_proportion}')
            if _current_proportion > target_proportion:
                raise MLSamplerException(f'Given target proportion ({target_proportion}) is smaller than current proportion ({_current_proportion})')
            _target_absolute_count: int = round((target_proportion / _current_proportion) * self.df.shape[0]) - _current_count
            _sample_idx: list = random.sample(population=list(self.df.loc[self.df[self.target] == target_class_value, :]), k=_target_absolute_count)
            _df: pd.DataFrame = pd.concat(objs=[self.df, self.df.iloc[_sample_idx, :]], axis=0)
            _new_current_count, _new_current_proportion = self._get_freq_of_target_class_value(df=_df,
                                                                                               target_class_value=target_class_value
                                                                                               )
            Log().log(msg=f'Up-sample target feature: {self.target} -> target class value: {target_class_value} -> new absolute frequency = {_new_current_count}, new relative frequency = {_new_current_proportion}, requested relative frequency = {target_proportion}')
            return _df
        else:
            raise MLSamplerException(f'Given target proportion ({target_proportion}) is invalid')


class SamplerException(Exception):
    """
    Class for handling exceptions for class Sampler
    """
    pass


class Sampler:
    """
    Class for general sampling purposes
    """
    def __init__(self, df, size: int = None, prop: float = None, **kwargs):
        """
        :param df: Pandas DataFrame
            Data set

        :param size: int
            Sample size

        :param prop: float
            Sample proportion

        :param kwargs: dict
            Key-word arguments for handling dask parameter settings
        """
        self.df: pd.DataFrame = df
        if size is None:
            if prop is None:
                raise SamplerException('Neither size nor proportion found')
            self.k: int = round(self.df.shape[0] * prop)
        else:
            self.k: int = size

    def random(self) -> pd.DataFrame:
        """
        Draw random sampling

        :return pd.DataFrame:
            Randomly sampled data set
        """
        _sample_idx: list = random.sample(population=list(self.df.index.values), k=self.k)
        Log().log(msg=f'Random sampling: Cases={self.k} ({round((self.k / self.df.shape[0]) * 100)} %)')
        return self.df.loc[_sample_idx, :]

    def quota(self, features: List[str], quotas: Dict[str, Dict[str, float]] = None) -> pd.DataFrame:
        """
        Draw quota sampling

        :param features: List[str]
            List of string containing the name of the features

        :param quotas: Dict[str, Dict[str, float]]
            Dictionary containing the quotas for each given feature

        :return pd.DataFrame:
            Sampled data set
        """
        _target_idx: List[int] = []
        for ft in features:
            if quotas is None:
                _counts: pd.Series = self.df[ft].value_counts(normalize=True)
            else:
                if ft not in quotas.keys():
                    raise SamplerException('Feature not found in user defined quotas settings')
                _counts: pd.Series = pd.Series(data=quotas.get(ft))
            _size: List[int] = [int(round(s, ndigits=0)) for s in _counts.values * self.k]
            for i, val in enumerate(_counts.index.values):
                _sample_idx: list = random.sample(population=self.df.loc[self.df[ft] == val, ft].index.get_values().tolist(), k=_size[i])
                _target_idx = _target_idx + _sample_idx
        return self.df.loc[list(set(_target_idx)), :]
