"""

Feature engineering of structured (tabular) data used in supervised machine learning

"""

import numpy as np
import pandas as pd
import random
import string

from custom_logger import Log
from sklearn.preprocessing import MinMaxScaler, Normalizer, RobustScaler, StandardScaler
from typing import Dict, List, Tuple


ENGINEERING_METH: Dict[str, List[str]] = dict(categorical=['one_hot_encoder'],
                                              continuous=['add',
                                                          'divide',
                                                          'exp_transform',
                                                          'log_transform',
                                                          'min_max_scaler',
                                                          'multiply',
                                                          'normalizer',
                                                          'power_transform',
                                                          'robust_scaler',
                                                          'square_root_transform',
                                                          'standard_scaler',
                                                          'subtract'
                                                          ],
                                              date=['date_categorizer',
                                                    'disparity'
                                                    ]
                                              )
ENGINEERING_METH.update({'ordinal': ENGINEERING_METH.get('categorical')})
ENGINEERING_METH['ordinal'].extend(ENGINEERING_METH.get('continuous'))
MIN_FEATURES_BY_METH: Dict[str, int] = dict(add=2,
                                            divide=2,
                                            multiply=2,
                                            subtract=2,
                                            disparity=2
                                            )


class FeatureEngineerException(Exception):
    """
    Class for handling exceptions for class FeatureEngineer
    """
    pass


class FeatureEngineer:
    """
    Class for feature engineering
    """
    def __init__(self, df: pd.DataFrame, processing_memory: dict = None):
        """
        :param df: pd.DataFrame
            Data set

        :param processing_memory: dict
            Processing memory
        """
        self.df: pd.DataFrame = df
        if processing_memory is None:
            self.processing_memory: dict = dict(level={'0': df.columns.tolist()},
                                                processor={},
                                                numeric_features=[],
                                                categorical_features=[],
                                                exclude=[]
                                                )
        else:
            self.processing_memory: dict = processing_memory

    def _force_rename_feature(self, feature: str, max_length: int = 100, new_length: int = 25) -> str:
        """
        Force feature renaming to avoid too long feature names

        :param feature: str
            Name of the feature to rename

        :param max_length: int
            Maximum character length of the feature

        :param new_length: int
            Length of the new randomly generated feature name

        :return: str
            Renamed feature name
        """
        if len(feature) > max_length:
            _character_pool: str = f'{string.ascii_lowercase}{string.digits}'
            _feature: str = ''.join(random.choices(population=_character_pool, k=new_length))
            while _feature in self.df.columns.tolist():
                _feature: str = ''.join(random.choices(population=_character_pool, k=new_length))
            Log().log(msg=f'Rename feature {feature} to {_feature}')
            return _feature
        else:
            return feature

    def add(self, feature_name: str, interaction_feature_name: str) -> np.ndarray:
        """
        Addition of two features

        :param feature_name: str
            Name of the feature to process

        :param interaction_feature_name: str
            Name of the feature to interact

        :return: np.ndarray
            Additive feature
        """
        return (self.df[feature_name] + self.df[interaction_feature_name]).values

    def date_categorizer(self,
                         feature_name: str,
                         year: bool = True,
                         month: bool = True,
                         week: bool = True,
                         week_day: bool = True,
                         day: bool = True,
                         hour: bool = True,
                         minute: bool = True,
                         second: bool = True
                         ) -> np.ndarray:
        """
        Extract categorical features based on datetime feature

        :param feature_name: str
            Name of the feature

        :param year: bool
            Extract year from date

        :param month: bool
            Extract month from date

        :param week: bool
            Extract week number of the year from date

        :param week_day: bool
            Extract week day from date

        :param day: bool
            Extract day from date

        :param hour: bool
            Extract hour from date

        :param minute: bool
            Extract minute from date

        :param second: bool
            Extract second from date

        :return: np.ndarray
            Categorized feature
        """
        if year:
            return self.df[feature_name].dt.year.values
        if month:
            return self.df[feature_name].dt.month.values
        if week:
            return self.df[feature_name].dt.week.values
        if week_day:
            return self.df[feature_name].dt.day_name().values
        if day:
            return self.df[feature_name].dt.day.values
        if hour:
            return self.df[feature_name].dt.hour.values
        if minute:
            return self.df[feature_name].dt.minute.values
        if second:
            return self.df[feature_name].dt.second.values
        raise FeatureEngineerException('No date categorizer parameter set to True')

    def disparity(self,
                  feature_name: str,
                  interaction_feature_name: str,
                  years: bool = True,
                  months: bool = True,
                  weeks: bool = True,
                  days: bool = True,
                  hours: bool = True,
                  minutes: bool = True,
                  seconds: bool = True,
                  digits: int = 6
                  ):
        """
        Calculate disparity time features

        :param feature_name: str
            Name of the feature

        :param interaction_feature_name: str
            Name of the feature to interact

        :param years: bool
            Whether to generate yearly differences between date features or not

        :param months: bool
            Whether to generate monthly differences between date features or not

        :param weeks: bool
            Whether to generate weekly differences between date features or not

        :param days: bool
            Whether to generate daily differences between date features or not

        :param hours: bool
            Whether to generate hourly differences between date features or not

        :param minutes: bool
            Whether to generate minutely differences between date features or not

        :param seconds: bool
            Whether to generate secondly differences between date features or not

        :param digits: int
            Amount of digits to round
        """
        if years:
            return np.round(a=((self.df[feature_name] - self.df[interaction_feature_name]).dt.days / 365).values, decimals=digits)
        if months:
            return np.round(a=((self.df[feature_name] - self.df[interaction_feature_name]).dt.days / 12).values, decimals=digits)
        if weeks:
            return np.round(a=((self.df[feature_name] - self.df[interaction_feature_name]).dt.days / 7).values, decimals=digits)
        if days:
            return np.round(a=(self.df[feature_name] - self.df[interaction_feature_name]).values, decimals=digits)
        if hours:
            return ((self.df[feature_name] - self.df[interaction_feature_name]).dt.days * 24).values
        if minutes:
            return (((self.df[feature_name] - self.df[interaction_feature_name]).dt.days * 24) * 60).values
        if seconds:
            return ((((self.df[feature_name] - self.df[interaction_feature_name]).dt.days * 24) * 60) * 60).values
        raise FeatureEngineerException('No date disparity parameter set to True')

    def disparity_time_series(self,
                              feature_name: str,
                              perc: bool = True,
                              by_col: bool = True,
                              imp_const: float = 0.000001,
                              periods: int = 1
                              ):
        """
        Calculate disparity for each time unit within time series

        :param feature_name: str
            Names of the feature

        :param perc: bool
            Calculate relative or absolute differences

        :param by_col: bool
            Calculate differences by column or row

        :param imp_const: float
            Constant value to impute missing values before calculating disparity

        :param periods: int
            Number of periods to use for calculation
        """
        _axis: int = 1 if by_col else 0
        _periods: int = 1 if periods < 1 else periods
        if perc:
            return self.df[feature_name].fillna(imp_const).pct_change(axis=_axis).fillna(0)
        else:
            return self.df[feature_name].fillna(imp_const).diff(periods=_periods, axis=_axis).fillna(0)

    def divide(self, feature_name: str, interaction_feature_name: str) -> np.ndarray:
        """
        Division of two features

        :param feature_name: str
            Name of the feature to process

        :param interaction_feature_name: str
            Name of the feature to interact

        :return: np.ndarray
            Divisive feature
        """
        return (self.df[feature_name] / self.df[interaction_feature_name]).values

    def exp_transform(self, feature_name: str) -> np.ndarray:
        """
        Exponential transformation

        :param feature_name: str
            Name of the feature to process

        :return: np.ndarray
            Exponential transformed feature
        """
        return np.exp(self.df[feature_name].values)

    def label_encoder(self, feature_name: str, encode: bool, encoder: dict = None) -> Tuple[np.ndarray, dict]:
        """
        Encode labels (written categories) into integer values

        :param feature_name: str
            Name of the feature to process

        :param encode: bool
            Encode labels into integers or decode integers into labels

        :param encoder: dict
            Mapping template for decoding integer to labels
        """
        if encode:
            _values: dict = {label: i for i, label in enumerate(self.df[feature_name].unique())}
        else:
            _data: pd.DataFrame = self.df[feature_name].replace({val: label for label, val in encoder})

    def log_transform(self, feature_name: str) -> np.ndarray:
        """
        Logarithmic transformation

        :param feature_name: str
            Name of the feature to process

        :return: np.ndarray
            Logarithmic transformed feature
        """
        return np.log(self.df[feature_name].values)

    def min_max_scaler(self, feature_name: str, minmax_range: Tuple[int, int] = (0, 1)) -> Tuple[np.ndarray, MinMaxScaler]:
        """
        Min-Max scaling

        :param feature_name: str
            Name of the feature to process

        :param minmax_range: Tuple[int, int]
            Range of allowed values

        :return: Tuple[np.ndarray, MinMaxScaler]
            Scaled feature and MinMaxScaler object
        """
        _minmax: MinMaxScaler = MinMaxScaler(feature_range=minmax_range)
        _minmax.fit(np.reshape(self.df[feature_name], (-1, 1)), y=None)
        return np.reshape(_minmax.transform(X=np.reshape(self.df[feature_name], (-1, 1))), (1, -1))[0], _minmax

    def multiply(self, feature_name: str, interaction_feature_name: str) -> np.ndarray:
        """
        Multiplication of two features

        :param feature_name: str
            Name of the feature to process

        :param interaction_feature_name: str
            Name of the feature to interact

        :return: np.ndarray
            Multiplicative feature
        """
        return (self.df[feature_name] * self.df[interaction_feature_name]).values

    def normalizer(self, feature_name: str, norm_meth: str = 'l2') -> Tuple[np.ndarray, Normalizer]:
        """
        Normalization

        :param feature_name: str
            Name of the feature to process

        :param norm_meth: str
            Abbreviated name of the used method for regularization
                -> l1: L1
                -> l2: L2

        :return: Tuple[np.ndarray, Normalizer]
            Normalized feature and Normalizer object
        """
        _normalizer: Normalizer = Normalizer(norm=norm_meth)
        _normalizer.fit(X=np.reshape(self.df[feature_name], (-1, 1)))
        return np.reshape(_normalizer.transform(X=np.reshape(self.df[feature_name], (-1, 1))), (1, -1))[0], _normalizer

    def one_hot_encoder(self, feature_name: str) -> pd.DataFrame:
        """
        One-hot encoding of categorical feature

        :param feature_name: str
            Name of the feature to process

        :return: pd.DataFrame
            One-hot encoded features
        """
        _dummies: pd.DataFrame = pd.get_dummies(data=self.df[feature_name],
                                                prefix=feature_name,
                                                prefix_sep='_',
                                                dummy_na=True,
                                                columns=None,
                                                sparse=False,
                                                drop_first=False,
                                                dtype=np.int64
                                                )
        _dummies = _dummies.loc[:, ~_dummies.columns.duplicated()]
        return _dummies

    def one_hot_merger(self, feature_name: str, interaction_feature_name: str) -> np.ndarray:
        """
        Merge one-hot encoded categorical features

        :param feature_name: str
            Name of the feature to process

        :param interaction_feature_name: str
            Name of the feature to interact

        :return: np.ndarray
            Merged one-hot encoded features
        """
        _add_one_hot_encoded_features: np.ndarray = self.df[feature_name].values + self.df[interaction_feature_name].values
        _add_one_hot_encoded_features[_add_one_hot_encoded_features > 1] = 1
        return _add_one_hot_encoded_features

    def power_transform(self, feature_name: str, exponent: int = 2) -> np.ndarray:
        """
        Transform continuous features using power transformation

        :param feature_name: str
            Name of the feature

        :param exponent: int
            Exponent value

        :return: np.ndarray
            Power transformed feature
        """
        return np.power(self.df[feature_name].values, exponent)

    def robust_scaler(self,
                      feature_name: str,
                      with_centering: bool = True,
                      with_scaling: bool = True,
                      quantile_range: Tuple[float, float] = (0.25, 0.75),
                      ) -> Tuple[np.ndarray, RobustScaler]:
        """
        Robust scaling of continuous feature

        :param feature_name: str
            Name of the feature to process

        :param with_centering: bool
            Use centering using robust scaler

        :param with_scaling: bool
            Use scaling using robust scaler

        :param quantile_range: Tuple[float, float]
            Quantile ranges of the robust scaler

        :return: Tuple[np.ndarray, RobustScaler]
            Scaled feature and RobustScaler object
        """
        _robust: RobustScaler = RobustScaler(with_centering=with_centering, with_scaling=with_scaling, quantile_range=quantile_range)
        _robust.fit(np.reshape(self.df[feature_name], (-1, 1)), y=None)
        return np.reshape(_robust.transform(X=np.reshape(self.df[feature_name], (-1, 1))), (1, -1))[0], _robust

    def square_root_transform(self, feature_name: str) -> np.ndarray:
        """
        Transform continuous features using square-root transformation

        :param feature_name: str
            Name of the feature

        :return np.ndarray
            Square root transformed feature
        """
        return np.square(self.df[feature_name].values)

    def standard_scaler(self,
                        feature_name: str,
                        with_mean: bool = True,
                        with_std: bool = True
                        ) -> Tuple[np.ndarray, StandardScaler]:
        """
        Standardize feature

        :param feature_name: str
            Name of the feature to process

        :param with_mean: bool
            Using mean to standardize features

        :param with_std: bool
            Using standard deviation to standardize features

        :return: Tuple[np.ndarray, StandardScaler]
            Scaled feature and StandardScaler object
        """
        _standard: StandardScaler = StandardScaler(with_mean=with_mean, with_std=with_std)
        _standard.fit(np.reshape(self.df[feature_name], (-1, 1)), y=None)
        return np.reshape(_standard.transform(X=np.reshape(self.df[feature_name], (-1, 1))), (1, -1))[0], _standard

    def subtract(self, feature_name: str, interaction_feature_name: str) -> np.ndarray:
        """
        Subtraction of two features

        :param feature_name: str
            Name of the feature to process

        :param interaction_feature_name: str
            Name of the feature to interact

        :return: np.ndarray
            Subtractive feature
        """
        return (self.df[feature_name] - self.df[interaction_feature_name]).values

    def main(self, feature_engineering_config: Dict[str, list]) -> pd.DataFrame:
        """
        Apply feature engineering using (tabular) structured data

        :param feature_engineering_config: Dict[str, list]
            Pre-defined configuration

        :return: pd.DataFrame
            Engineered data set
        """
        _level: int = 0
        while self.processing_memory['level'].get(str(_level)) is not None:
            _level += 1
        self.processing_memory['level'].update({str(_level): {}})
        _df: pd.DataFrame = pd.DataFrame()
        for meth in feature_engineering_config.keys():
            for element in feature_engineering_config[meth]:
                if isinstance(element, str):
                    if meth.find('exp') >= 0:
                        _new_feature_name: str = self._force_rename_feature(feature=f'{element}_exp')
                        _df[_new_feature_name] = self.exp_transform(feature_name=element)
                        self.processing_memory['numeric_features'].append(_new_feature_name)
                        self.processing_memory['level'][str(_level)].update({_new_feature_name: dict(meth='exp_transform',
                                                                                                     param=None,
                                                                                                     feature=element,
                                                                                                     interactor=None
                                                                                                     )
                                                                             })
                        Log().log(msg=f'Generated feature "{_new_feature_name}": transformed feature "{element}" using exponential transform method')
                    elif meth.find('log') >= 0:
                        _new_feature_name: str = self._force_rename_feature(feature=f'{element}_log')
                        _df[_new_feature_name] = self.log_transform(feature_name=element)
                        self.processing_memory['numeric_features'].append(_new_feature_name)
                        self.processing_memory['level'][str(_level)].update({_new_feature_name: dict(meth='log_transform',
                                                                                                     param=None,
                                                                                                     feature=element,
                                                                                                     interactor=None
                                                                                                     )
                                                                             })
                        Log().log(msg=f'Generated feature "{_new_feature_name}": transformed feature "{element}" using logarithmic transform method')
                    elif meth.find('min_max') >= 0:
                        _new_feature_name: str = self._force_rename_feature(feature=f'{element}_minmax')
                        _new_feature, _scaler_obj = self.min_max_scaler(feature_name=element,
                                                                        minmax_range=(0, 1)
                                                                        )
                        _df[_new_feature_name] = _new_feature
                        self.processing_memory['numeric_features'].append(_new_feature_name)
                        self.processing_memory['processor'].update({_new_feature_name: {'min_max_scaler': _scaler_obj}})
                        self.processing_memory['level'][str(_level)].update({_new_feature_name: dict(meth='min_max_scaler',
                                                                                                     param=None,
                                                                                                     feature=element,
                                                                                                     interactor=None
                                                                                                     )
                                                                             })
                        Log().log(msg=f'Generated feature "{_new_feature_name}": transformed feature "{element}" using min-max scaling method')
                    elif meth.find('norm') >= 0:
                        _new_feature_name: str = self._force_rename_feature(feature=f'{element}_norm')
                        _new_feature, _scaler_obj = self.normalizer(feature_name=element,
                                                                    norm_meth='l2'
                                                                    )
                        _df[_new_feature_name] = _new_feature
                        self.processing_memory['numeric_features'].append(_new_feature_name)
                        self.processing_memory['processor'].update({_new_feature_name: {'norm': _scaler_obj}})
                        self.processing_memory['level'][str(_level)].update({_new_feature_name: dict(meth='normalizer',
                                                                                                     param=None,
                                                                                                     feature=element,
                                                                                                     interactor=None
                                                                                                     )
                                                                             })
                        Log().log(msg=f'Generated feature "{_new_feature_name}": transformed feature "{element}" using normalizing scaling method')
                    elif meth == 'one_hot_encoder':
                        _df_one_hot: pd.DataFrame = self.one_hot_encoder(feature_name=element)
                        _df = pd.concat(objs=[_df, _df_one_hot])
                        for new_feature_name in _df_one_hot.columns.tolist():
                            self.processing_memory['categorical_features'].append(new_feature_name)
                            self.processing_memory['level'][str(_level)].update({new_feature_name: dict(meth='one_hot_encoder',
                                                                                                        param=None,
                                                                                                        feature=element,
                                                                                                        interactor=None
                                                                                                        )
                                                                                 })
                            Log().log(msg=f'Generated feature "{new_feature_name}": transformed feature "{element}" using one-hot encoding method')
                    elif meth.find('pow') >= 0:
                        _new_feature_name: str = self._force_rename_feature(feature=f'{element}_pow')
                        _df[_new_feature_name] = self.power_transform(feature_name=element,
                                                                      exponent=2
                                                                      )
                        self.processing_memory['numeric_features'].append(_new_feature_name)
                        self.processing_memory['level'][str(_level)].update({_new_feature_name: dict(meth='power_transform',
                                                                                                     param=None,
                                                                                                     feature=element,
                                                                                                     interactor=None
                                                                                                     )
                                                                             })
                        Log().log(msg=f'Generated feature "{_new_feature_name}": transformed feature "{element}" using power transform method')
                    elif meth.find('robust') >= 0:
                        _new_feature_name: str = self._force_rename_feature(feature=f'{element}_robust')
                        _new_feature, _scaler_obj = self.robust_scaler(feature_name=element,
                                                                       with_centering=True,
                                                                       with_scaling=True,
                                                                       quantile_range=(0.25, 0.75)
                                                                       )
                        _df[_new_feature_name] = _new_feature
                        self.processing_memory['numeric_features'].append(_new_feature_name)
                        self.processing_memory['processor'].update({_new_feature_name: {'robust': _scaler_obj}})
                        self.processing_memory['level'][str(_level)].update({_new_feature_name: dict(meth='robust_scaler',
                                                                                                     param=None,
                                                                                                     feature=element,
                                                                                                     interactor=None
                                                                                                     )
                                                                             })
                        Log().log(msg=f'Generated feature "{_new_feature_name}": transformed feature "{element}" using robust scaling method')
                    elif meth.find('square') >= 0:
                        _new_feature_name: str = self._force_rename_feature(feature=f'{element}_squared')
                        _df[_new_feature_name] = self.square_root_transform(feature_name=element)
                        self.processing_memory['numeric_features'].append(_new_feature_name)
                        self.processing_memory['level'][str(_level)].update({_new_feature_name: dict(meth='square_root_transform',
                                                                                                     param=None,
                                                                                                     feature=element,
                                                                                                     interactor=None
                                                                                                     )
                                                                             })
                        Log().log(msg=f'Generated feature "{_new_feature_name}": transformed feature "{element}" using square root transform method')
                    elif meth.find('standard') >= 0:
                        _new_feature_name: str = self._force_rename_feature(feature=f'{element}_standard')
                        _new_feature, _scaler_obj = self.standard_scaler(feature_name=element,
                                                                         with_mean=True,
                                                                         with_std=True
                                                                         )
                        _df[_new_feature_name] = _new_feature
                        self.processing_memory['numeric_features'].append(_new_feature_name)
                        self.processing_memory['processor'].update({_new_feature_name: {'standard': _scaler_obj}})
                        self.processing_memory['level'][str(_level)].update({_new_feature_name: dict(meth='standard_scaler',
                                                                                                     param=None,
                                                                                                     feature=element,
                                                                                                     interactor=None
                                                                                                     )
                                                                             })
                        Log().log(msg=f'Generated feature "{_new_feature_name}": transformed feature "{element}" using standard scaling method')
                    else:
                        raise FeatureEngineerException(f'Feature engineering method ({meth}) not supported')
                elif isinstance(element, tuple):
                    if meth.find('add') >= 0:
                        _new_feature_name: str = self._force_rename_feature(feature=f'{element[0]}_add_{element[1]}')
                        _df[_new_feature_name] = self.add(feature_name=element[0],
                                                          interaction_feature_name=element[1]
                                                          )
                        self.processing_memory['numeric_features'].append(_new_feature_name)
                        self.processing_memory['level'][str(_level)].update({_new_feature_name: dict(meth='add',
                                                                                                     param=None,
                                                                                                     feature=element[0],
                                                                                                     interactor=element[1]
                                                                                                     )
                                                                             })
                        Log().log(msg=f'Generated feature "{_new_feature_name}": addition of feature "{element[0]}" and "{element[1]}"')
                    elif meth.find('date_categorizer') >= 0:
                        try:
                            assert pd.to_datetime(self.df[element])
                        except (ValueError, TypeError):
                            Log().log(msg=f'Feature "{element}" could not be converted to datetime')
                            continue
                        _new_feature_name: str = self._force_rename_feature(feature=f'{element[0]}_{element[1]}')
                        _df[_new_feature_name] = self.date_categorizer(feature_name=element[0],
                                                                       year=True if element[1] == 'year' else False,
                                                                       month=True if element[1] == 'month' else False,
                                                                       week=True if element[1] == 'week' else False,
                                                                       week_day=True if element[1] == 'week_day' else False,
                                                                       day=True if element[1] == 'day' else False,
                                                                       hour=True if element[1] == 'hour' else False,
                                                                       minute=True if element[1] == 'minute' else False,
                                                                       second=True if element[1] == 'second' else False
                                                                       )
                        if _new_feature_name not in self.processing_memory['exclude']:
                            self.processing_memory['exclude'].append(_new_feature_name)
                        self.processing_memory['level'][str(_level)].update({_new_feature_name: dict(meth='date_categorizer',
                                                                                                     param=dict(action=element[1]),
                                                                                                     feature=element[0],
                                                                                                     interactor=None
                                                                                                     )
                                                                             })
                        _df_one_hot: pd.DataFrame = self.one_hot_encoder(feature_name=_new_feature_name)
                        _df = pd.concat(objs=[_df, _df_one_hot])
                        for new_feature_name in _df_one_hot.columns.tolist():
                            self.processing_memory['categorical_features'].append(new_feature_name)
                            self.processing_memory['level'][str(_level)].update({new_feature_name: dict(meth='one_hot_encoder',
                                                                                                        param=None,
                                                                                                        feature=_new_feature_name,
                                                                                                        interactor=None
                                                                                                        )
                                                                                 })
                            Log().log(msg=f'Generated feature "{new_feature_name}": transformed feature "{_new_feature_name}" using one-hot encoding method')
                    elif meth.find('disparity') >= 0:
                        try:
                            assert pd.to_datetime(self.df[element[0]])
                        except (ValueError, TypeError):
                            Log().log(msg=f'Feature "{element[0]}" could not be converted to datetime')
                            continue
                        try:
                            assert pd.to_datetime(self.df[element[1]])
                        except (ValueError, TypeError):
                            Log().log(msg=f'Feature "{element[1]}" could not be converted to datetime')
                            continue
                        _new_feature_name: str = self._force_rename_feature(feature=f'time_diff_{element[0]}_{element[1]}_in_{element[2]}')
                        _df[_new_feature_name] = self.disparity(feature_name=element[0],
                                                                interaction_feature_name=element[1],
                                                                years=True if element[2] == 'years' else False,
                                                                months=True if element[2] == 'months' else False,
                                                                weeks=True if element[2] == 'weeks' else False,
                                                                days=True if element[2] == 'days' else False,
                                                                hours=True if element[2] == 'hours' else False,
                                                                minutes=True if element[2] == 'minutes' else False,
                                                                seconds=True if element[2] == 'seconds' else False
                                                                )
                        self.processing_memory['numeric_features'].append(_new_feature_name)
                        self.processing_memory['level'][str(_level)].update({_new_feature_name: dict(meth='disparity',
                                                                                                     param=dict(action=element[2]),
                                                                                                     feature=element[0],
                                                                                                     interactor=element[1]
                                                                                                     )
                                                                             })
                        Log().log(msg=f'Generated feature "{_new_feature_name}": time disparity of feature "{element[0]}" and "{element[1]}" in "{element[2]}"')
                    elif meth.find('div') >= 0:
                        _new_feature_name: str = self._force_rename_feature(feature=f'{element[0]}_div_{element[1]}')
                        _df[_new_feature_name] = self.divide(feature_name=element[0],
                                                             interaction_feature_name=element[1]
                                                             )
                        self.processing_memory['numeric_features'].append(_new_feature_name)
                        self.processing_memory['level'][str(_level)].update({_new_feature_name: dict(meth='divide',
                                                                                                     param=None,
                                                                                                     feature=element[0],
                                                                                                     interactor=element[1]
                                                                                                     )
                                                                             })
                        Log().log(msg=f'Generated feature "{_new_feature_name}": division of feature "{element[0]}" and "{element[1]}"')
                    elif meth.find('multi') >= 0:
                        _new_feature_name: str = self._force_rename_feature(feature=f'{element[0]}_multi_{element[1]}')
                        _df[_new_feature_name] = self.multiply(feature_name=element[0],
                                                               interaction_feature_name=element[1]
                                                               )
                        self.processing_memory['numeric_features'].append(_new_feature_name)
                        self.processing_memory['level'][str(_level)].update({_new_feature_name: dict(meth='multiply',
                                                                                                     param=None,
                                                                                                     feature=element[0],
                                                                                                     interactor=element[1]
                                                                                                     )
                                                                             })
                        Log().log(msg=f'Generated feature "{_new_feature_name}": multiplication of feature "{element[0]}" and "{element[1]}"')
                    elif meth == 'one_hot_merger':
                        _new_feature_name: str = self._force_rename_feature(feature=f'{element[0]}_m_{element[1]}')
                        _df[_new_feature_name] = self.one_hot_merger(feature_name=element[0],
                                                                     interaction_feature_name=element[1]
                                                                     )
                        self.processing_memory['categorical_features'].append(_new_feature_name)
                        self.processing_memory['level'][str(_level)].update({_new_feature_name: dict(meth='one_hot_merger',
                                                                                                     param=None,
                                                                                                     feature=element[0],
                                                                                                     interactor=element[1]
                                                                                                     )
                                                                             })
                        Log().log(msg=f'Generated feature "{_new_feature_name}": merging of one-hot encoded feature "{element[0]}" and "{element[1]}"')
                    elif meth.find('sub') >= 0:
                        _new_feature_name: str = self._force_rename_feature(feature=f'{element[0]}_sub_{element[1]}')
                        _df[_new_feature_name] = self.subtract(feature_name=element[0],
                                                               interaction_feature_name=element[1]
                                                               )
                        self.processing_memory['numeric_features'].append(_new_feature_name)
                        self.processing_memory['level'][str(_level)].update({_new_feature_name: dict(meth='subtract',
                                                                                                     param=None,
                                                                                                     feature=element[0],
                                                                                                     interactor=element[1]
                                                                                                     )
                                                                             })
                        Log().log(msg=f'Generated feature "{_new_feature_name}": subtraction of feature "{element[0]}" and "{element[1]}"')
                    else:
                        raise FeatureEngineerException(f'Feature engineering method ({meth}) not supported')
                else:
                    raise FeatureEngineerException(f'Element type ({element}) not supported')
        return _df

    def re_engineering(self, features: List[str]) -> pd.DataFrame:
        """
        Re-engineer features for inference

        :param features: List[str]
            Names of features to re-engineer

        :return: pd.DataFrame
            Re-engineered data set
        """
        pass
