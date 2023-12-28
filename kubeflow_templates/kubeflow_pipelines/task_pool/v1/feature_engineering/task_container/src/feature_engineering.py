"""

Feature engineering of structured (tabular) data used in supervised machine learning

"""

import copy
import numpy as np
import pandas as pd
import random
import string

from custom_logger import Log
from sklearn.preprocessing import MinMaxScaler, Normalizer, RobustScaler, StandardScaler
from typing import Dict, List, Tuple


ENGINEERING_METH: Dict[str, List[str]] = dict(categorical=['one_hot_encoder',
                                                           'one_hot_merger'
                                                           ],
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
ENGINEERING_NUMERIC_INTERACTION_METH: List[str] = ['add',
                                                   'divide',
                                                   'multiply',
                                                   'subtract'
                                                   ]
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
    def __init__(self,
                 df: pd.DataFrame,
                 analytical_data_types: Dict[str, List[str]],
                 features: List[str] = None,
                 processing_memory: dict = None,
                 feature_engineering_config: Dict[str, list] = None
                 ):
        """
        :param df: pd.DataFrame
            Data set

        :param analytical_data_types: Dict[str, List[str]]
            Analytical data types assignment

        :param features: List[str]
            Name of the features

        :param processing_memory: dict
            Processing memory

        :param feature_engineering_config: Dict[str, list]
            Pre-defined configuration
        """
        self.df: pd.DataFrame = df
        self.features: List[str] = self.df.columns.tolist() if features is None else features
        if processing_memory is None:
            self.processing_memory: dict = dict(level={'0': df.columns.tolist()},
                                                processor={},
                                                feature_relations={},
                                                analytical_data_types=analytical_data_types,
                                                next_level_numeric_features_base=[],
                                                next_level_categorical_features_base=[],
                                                numeric_features=[],
                                                categorical_features=[],
                                                exclude=[]
                                                )
        else:
            self.processing_memory: dict = processing_memory
        self.level: int = len(self.processing_memory['level'].keys())
        self.processor = None
        self.feature_engineering_config: Dict[str, list] = {} if feature_engineering_config is None else feature_engineering_config
        if self.feature_engineering_config is None:
            self._config_feature_engineering()

    def _config_feature_engineering(self) -> None:
        """
        Configure feature engineering based on processor memory
        """
        if len(self.processing_memory['level'].keys()) == 1:
            for analytical_data_type in self.processing_memory['analytical_data_types'].keys():
                _n_features: int = len(self.processing_memory['analytical_data_types'].get(analytical_data_type))
                if _n_features > 0:
                    for meth in ENGINEERING_METH.get(analytical_data_type):
                        _min_features: int = 1 if MIN_FEATURES_BY_METH.get(meth) is None else MIN_FEATURES_BY_METH.get(meth)
                        if _n_features >= _min_features:
                            self.feature_engineering_config.update({meth: []})
                            for feature in self.processing_memory['analytical_data_types'].get(analytical_data_type):
                                if _min_features == 1:
                                    if feature in self.features:
                                        self.feature_engineering_config[meth].append(feature)
                                else:
                                    for interactor in self.processing_memory['analytical_data_types'].get(analytical_data_type):
                                        if feature in self.features:
                                            if feature != interactor:
                                                if interactor in self.features:
                                                    self.feature_engineering_config[meth].append((feature, interactor))
        else:
            _next_level_numeric_features: List[str] = [numeric_feature for numeric_feature in self.processing_memory['next_level_numeric_features_base'] if numeric_feature in self.df.columns.tolist()]
            if len(_next_level_numeric_features) >= 2:
                _n_numeric_pairs: int = int(len(_next_level_numeric_features) / 2)
                if _n_numeric_pairs % 2 != 1:
                    _n_numeric_pairs += 1
                _numeric_pairs: List[np.array] = np.array_split(ary=np.array(_next_level_numeric_features), indices_or_sections=_n_numeric_pairs)
                for numeric_pair in _numeric_pairs:
                    if numeric_pair[1] in self.processing_memory['feature_relations'][numeric_pair[0]]:
                        continue
                    for meth in ENGINEERING_NUMERIC_INTERACTION_METH:
                        self.feature_engineering_config[meth].append((numeric_pair[0], numeric_pair[1]))
            _next_level_categorical_features: List[str] = [categorical_feature for categorical_feature in self.processing_memory['next_level_categorical_features_base'] if categorical_feature in self.df.columns.tolist()]
            if len(_next_level_categorical_features) >= 2:
                _n_categorical_pairs: int = int(len(_next_level_categorical_features) / 2)
                if _n_categorical_pairs % 2 != 1:
                    _n_categorical_pairs += 1
                _categorical_pairs: List[np.array] = np.array_split(ary=np.array(_next_level_categorical_features), indices_or_sections=_n_categorical_pairs)
                for categorical_pair in _categorical_pairs:
                    if categorical_pair[1] in self.processing_memory['feature_relations'][categorical_pair[0]]:
                        continue
                    self.feature_engineering_config['one_hot_merger'].append((categorical_pair[0], categorical_pair[1]))
        Log().log(msg=f'Configure feature engineering: {len(self.feature_engineering_config.keys())} actions')

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

    def _get_feature_relations(self, feature: str) -> List[str]:
        """
        Get feature relations of given feature

        :param feature: str
            Name of the feature

        :return: List[str]
            Name of the features related to given feature
        """
        _feature_relations: List[str] = []
        if self.processing_memory['feature_relations'].get(feature) is None:
            Log().log(msg=f'No feature relations for feature ({feature}) found')
        else:
            _direct_relations: List[str] = copy.deepcopy(self.processing_memory['feature_relations'][feature])
            _indirect_relations: List[str] = []
            for relation in _direct_relations:
                _keep_searching: bool = True
                _next_relations: List[str] = []
                while _keep_searching:
                    if self.processing_memory['feature_relations'].get(relation) is None:
                        _keep_searching = False
                    else:
                        if len(_next_relations) == 0:
                            _next_relations.append(copy.deepcopy(self.processing_memory['feature_relations'][relation]))
                            _indirect_relations.extend(copy.deepcopy(self.processing_memory['feature_relations'][relation]))
                        if self.processing_memory['feature_relations'].get(_next_relations[0]) is None:
                            del _next_relations[0]
                        else:
                            _next_relations.extend(copy.deepcopy(self.processing_memory['feature_relations'][_next_relations[0]]))
                            _indirect_relations.extend(copy.deepcopy(self.processing_memory['feature_relations'][_next_relations[0]]))
                    if len(_next_relations) == 0:
                        _keep_searching = False
            _direct_relations.extend(_indirect_relations)
            Log().log(msg=f'Found {len(_direct_relations)} relations for feature ({feature})')
        return _feature_relations

    def _process_memory(self, meth: str, param: dict, feature: str, interactor: str, new_feature: str, categorical: bool) -> None:
        """
        Process memory

        :param meth:
        :param param:
        :param feature:
        :param interactor:
        :param new_feature:
        :param categorical:
        """
        self.processing_memory['level'][str(self.level)].update({new_feature: dict(meth=meth,
                                                                                   param=param,
                                                                                   feature=feature,
                                                                                   interactor=interactor
                                                                                   )
                                                                 })
        if self.processor is not None:
            if self.processing_memory['processor'].get(new_feature) is None:
                self.processing_memory['processor'].update({new_feature: self.processor})
            else:
                self.processing_memory['processor'].update({new_feature: self.processor})
            self.processor = None
        if self.processing_memory['feature_relations'].get(new_feature) is None:
            self.processing_memory['feature_relations'].update({new_feature: [feature]})
        else:
            if feature not in self.processing_memory['feature_relations'][new_feature]:
                self.processing_memory['feature_relations'][new_feature].append(feature)
        if interactor is not None:
            if interactor not in self.processing_memory['feature_relations'][new_feature]:
                self.processing_memory['feature_relations'][new_feature].append(interactor)
        if categorical:
            self.processing_memory['analytical_data_types']['categorical'].append(new_feature)
            self.processing_memory['next_level_categorical_features_base'].append(new_feature)
        else:
            self.processing_memory['analytical_data_types']['continuous'].append(new_feature)
            self.processing_memory['next_level_numeric_features_base'].append(new_feature)

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
                         ) -> pd.DataFrame:
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

        :return: pd.DataFrame
            Categorized features
        """
        _df: pd.DataFrame = pd.DataFrame()
        if year:
            _df[f'{feature_name}_year'] = self.df[feature_name].dt.year.values
        if month:
            _df[f'{feature_name}_month'] = self.df[feature_name].dt.month.values
        if week:
            _df[f'{feature_name}_week'] = self.df[feature_name].dt.week.values
        if week_day:
            _df[f'{feature_name}_week_day'] = self.df[feature_name].dt.day_name().values
        if day:
            _df[f'{feature_name}_day'] = self.df[feature_name].dt.day.values
        if hour:
            _df[f'{feature_name}_hour'] = self.df[feature_name].dt.hour.values
        if minute:
            _df[f'{feature_name}_minute'] = self.df[feature_name].dt.minute.values
        if second:
            _df[f'{feature_name}_second'] = self.df[feature_name].dt.second.values
        return _df

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
                  ) -> pd.DataFrame:
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

        :return: pd.DataFrame
            Time disparity features
        """
        _df: pd.DataFrame = pd.DataFrame()
        if years:
            _df[f'{feature_name}_{interaction_feature_name}_disparity_years'] = np.round(a=((self.df[feature_name] - self.df[interaction_feature_name]).dt.days / 365).values, decimals=digits)
        if months:
            _df[f'{feature_name}_{interaction_feature_name}_disparity_months'] = np.round(a=((self.df[feature_name] - self.df[interaction_feature_name]).dt.days / 12).values, decimals=digits)
        if weeks:
            _df[f'{feature_name}_{interaction_feature_name}_disparity_weeks'] = np.round(a=((self.df[feature_name] - self.df[interaction_feature_name]).dt.days / 7).values, decimals=digits)
        if days:
            _df[f'{feature_name}_{interaction_feature_name}_disparity_days'] = np.round(a=(self.df[feature_name] - self.df[interaction_feature_name]).values, decimals=digits)
        if hours:
            _df[f'{feature_name}_{interaction_feature_name}_disparity_hours'] = ((self.df[feature_name] - self.df[interaction_feature_name]).dt.days * 24).values
        if minutes:
            _df[f'{feature_name}_{interaction_feature_name}_disparity_minutes'] = (((self.df[feature_name] - self.df[interaction_feature_name]).dt.days * 24) * 60).values
        if seconds:
            _df[f'{feature_name}_{interaction_feature_name}_disparity_seconds'] = ((((self.df[feature_name] - self.df[interaction_feature_name]).dt.days * 24) * 60) * 60).values
        return _df

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

    def generate_re_engineering_instructions(self, features: List[str]) -> List[dict]:
        """
        Generate instructions for re-engineering used in inference pipeline

        :param features: List[str]
            Name of the features

        :return: List[dict]
            Step-by-step engineering instructions
        """
        _instructions: List[dict] = []
        for feature in features:
            _feature_relations: List[str] = self._get_feature_relations(feature=feature)
            for relation in _feature_relations:
                for level in range(self.level - 1, 0, -1):
                    if self.processing_memory['level'][str(level)].get(relation) is not None:
                        _instructions.append(self.processing_memory['level'][str(level)][relation])
                        break
        return _instructions

    def label_decoder(self, feature_name: str) -> np.ndarray:
        """
        Decode numeric into verbal categories

        :param feature_name: str
            Name of the feature to process

        :return: np.ndarray
            Label decoded feature
        """
        _df: pd.DataFrame = pd.DataFrame()
        _encoding: dict = self.processing_memory['processor'][feature_name]
        _df[f'{feature_name}_label'] = self.df[feature_name].values
        _df[f'{feature_name}_label'].replace(to_replace={val: label for label, val in _encoding}, inplace=True)
        return _df[f'{feature_name}_label'].values

    def label_encoder(self, feature_name: str) -> np.ndarray:
        """
        Encode labels (verbal categories) into numeric values

        :param feature_name: str
            Name of the feature to process

        :return: np.ndarray
            label encoded feature
        """
        _df: pd.DataFrame = pd.DataFrame()
        self.processor = {label: i for i, label in enumerate(self.df[feature_name].unique())}
        _df[f'{feature_name}_enc'] = self.df[feature_name].values
        _df[f'{feature_name}_enc'].replace(to_replace=self.processor, inplace=True)
        return _df[f'{feature_name}_enc'].values

    def log_transform(self, feature_name: str) -> np.ndarray:
        """
        Logarithmic transformation

        :param feature_name: str
            Name of the feature to process

        :return: np.ndarray
            Logarithmic transformed feature
        """
        return np.log(self.df[feature_name].values)

    def main(self, feature_engineering_config: Dict[str, list]) -> pd.DataFrame:
        """
        Apply feature engineering using (tabular) structured data

        :return: pd.DataFrame
            Engineered data set
        """
        self.processing_memory['level'].update({str(self.level): {}})
        _df: pd.DataFrame = pd.DataFrame()
        for meth in feature_engineering_config.keys():
            _engineering_meth = getattr(self, meth, None)
            for element in feature_engineering_config[meth]:
                if isinstance(element, str):
                    _param: dict = dict(feature_name=element)
                elif isinstance(element, tuple):
                    _param: dict = dict(feature_name=element[0], interaction_feature_name=element[1])
                else:
                    raise FeatureEngineerException(f'Element type ({element}) not supported')
                if _engineering_meth and callable(_engineering_meth):
                    if meth == 'one_hot_encoder':
                        _df_one_hot: pd.DataFrame = _engineering_meth(**_param)
                        for new_feature_name in _df_one_hot.columns.tolist():
                            if new_feature_name not in _df.columns.tolist():
                                _new_feature_name: str = self._force_rename_feature(feature=new_feature_name)
                                _df[_new_feature_name] = _df_one_hot[new_feature_name].values
                                if self.processing_memory['processor'].get(_new_feature_name) is None:
                                    self.processing_memory['processor'].update({element: [_new_feature_name]})
                                else:
                                    self.processing_memory['processor'][element].append(_new_feature_name)
                                self._process_memory(meth=meth,
                                                     param=_param,
                                                     feature=_param.get('feature_name'),
                                                     interactor=_param.get('interaction_feature_name'),
                                                     new_feature=_new_feature_name,
                                                     categorical=True
                                                     )
                    elif meth == 'date_categorizer':
                        _df_date_categorized: pd.DataFrame = _engineering_meth(**_param)
                        for new_categorical_feature_name in _df_date_categorized.columns.tolist():
                            if new_categorical_feature_name not in _df.columns.tolist():
                                _new_feature_name: str = self._force_rename_feature(feature=new_categorical_feature_name)
                                _df[_new_feature_name] = _df_date_categorized[new_categorical_feature_name].values
                                self._process_memory(meth=meth,
                                                     param=_param,
                                                     feature=_param.get('feature_name'),
                                                     interactor=_param.get('interaction_feature_name'),
                                                     new_feature=_new_feature_name,
                                                     categorical=True
                                                     )
                        _df_one_hot: pd.DataFrame = self.one_hot_encoder(feature_name=element)
                        for new_feature_name in _df_one_hot.columns.tolist():
                            if new_feature_name not in _df.columns.tolist():
                                _new_feature_name: str = self._force_rename_feature(feature=new_feature_name)
                                _df[_new_feature_name] = _df_one_hot[new_feature_name].values
                                if self.processing_memory['processor'].get(_new_feature_name) is None:
                                    self.processing_memory['processor'].update({element: [_new_feature_name]})
                                else:
                                    self.processing_memory['processor'][element].append(_new_feature_name)
                                self._process_memory(meth='one_hot_encoder',
                                                     param=_param,
                                                     feature=_param.get('feature_name'),
                                                     interactor=_param.get('interaction_feature_name'),
                                                     new_feature=_new_feature_name,
                                                     categorical=True
                                                     )
                    elif meth == 'disparity':
                        _df_disparity: pd.DataFrame = _engineering_meth(**_param)
                        for new_feature_name in _df_disparity.columns.tolist():
                            if new_feature_name not in _df.columns.tolist():
                                _new_feature_name: str = self._force_rename_feature(feature=new_feature_name)
                                _df[_new_feature_name] = _df_disparity[new_feature_name].values
                                self._process_memory(meth=meth,
                                                     param=_param,
                                                     feature=_param.get('feature_name'),
                                                     interactor=_param.get('interaction_feature_name'),
                                                     new_feature=_new_feature_name,
                                                     categorical=False
                                                     )
                    else:
                        _new_feature_name: str = self._force_rename_feature(feature=f'{element}_{meth}')
                        _df[_new_feature_name] = _engineering_meth(**_param)
                        self._process_memory(meth=meth,
                                             param=_param,
                                             feature=_param.get('feature_name'),
                                             interactor=_param.get('interaction_feature_name'),
                                             new_feature=_new_feature_name,
                                             categorical=True if meth in ENGINEERING_METH.get('categorical') else False
                                             )
                        Log().log(msg=f'Generated feature "{_new_feature_name}": transformed feature "{element}" using "{meth}" method')
                else:
                    Log().log(msg=f'Feature engineering method ({meth}) not supported')
        return _df

    def min_max_scaler(self, feature_name: str, minmax_range: Tuple[int, int] = (0, 1)) -> np.ndarray:
        """
        Min-Max scaling

        :param feature_name: str
            Name of the feature to process

        :param minmax_range: Tuple[int, int]
            Range of allowed values

        :return: np.ndarray
            Min-Max scaled feature
        """
        _minmax: MinMaxScaler = MinMaxScaler(feature_range=minmax_range)
        _minmax.fit(np.reshape(self.df[feature_name], (-1, 1)), y=None)
        self.processor = _minmax
        return np.reshape(_minmax.transform(X=np.reshape(self.df[feature_name], (-1, 1))), (1, -1))[0]

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

    def normalizer(self, feature_name: str, norm_meth: str = 'l2') -> np.ndarray:
        """
        Normalization

        :param feature_name: str
            Name of the feature to process

        :param norm_meth: str
            Abbreviated name of the used method for regularization
                -> l1: L1
                -> l2: L2

        :return: np.ndarray
            Normalized feature
        """
        _normalizer: Normalizer = Normalizer(norm=norm_meth)
        _normalizer.fit(X=np.reshape(self.df[feature_name], (-1, 1)))
        self.processor = _normalizer
        return np.reshape(_normalizer.transform(X=np.reshape(self.df[feature_name], (-1, 1))), (1, -1))[0]

    def one_hot_decoder(self, feature_name: str) -> pd.DataFrame:
        """
        One-hot decoding of categorical feature for inference pipeline

        :param feature_name: str
            Name of the feature

        :return: pd.DataFrame
            One-hot encoded features
        """
        _df: pd.DataFrame = pd.DataFrame()
        _categorical_values: list = [str(val).split(sep=f'{feature_name}_')[-1] for val in self.processing_memory['processor'][feature_name]]
        for i, feature in enumerate(self.processing_memory['processor'][feature_name]):
            _df[feature] = self.df[feature_name].apply(lambda x: 1 if x == _categorical_values[i] else 0)
        return _df

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

    def re_engineering(self, features: List[str]) -> pd.DataFrame:
        """
        Re-engineer features for inference (batch-prediction)

        :param features: List[str]
            Names of features to re-engineer

        :return: pd.DataFrame
            Re-engineered data set
        """
        _df: pd.DataFrame = pd.DataFrame()
        for feature in features:
            _feature_relations: List[str] = self._get_feature_relations(feature=feature)
            for relation in _feature_relations:
                for level in range(self.level - 1, 0, -1):
                    if self.processing_memory['level'][str(level)].get(relation) is not None:
                        if self.processing_memory['level'][str(level)][relation]['meth'] == 'one_hot_encoder':
                            pass
                        elif self.processing_memory['level'][str(level)][relation]['meth'] == 'one_hot_merger':
                            pass
                        elif self.processing_memory['level'][str(level)][relation]['meth'] == 'add':
                            pass
                        elif self.processing_memory['level'][str(level)][relation]['meth'] == 'divide':
                            pass
                        elif self.processing_memory['level'][str(level)][relation]['meth'] == 'multiply':
                            pass
                        elif self.processing_memory['level'][str(level)][relation]['meth'] == 'subtract':
                            pass
                        elif self.processing_memory['level'][str(level)][relation]['meth'] == 'exp_transform':
                            pass
                        elif self.processing_memory['level'][str(level)][relation]['meth'] == 'log_transform':
                            pass
                        elif self.processing_memory['level'][str(level)][relation]['meth'] == 'add':
                            pass
                        elif self.processing_memory['level'][str(level)][relation]['meth'] == 'add':
                            pass
                        elif self.processing_memory['level'][str(level)][relation]['meth'] == 'add':
                            pass
                        break
        return _df

    def re_scaler(self, feature_name: str) -> np.ndarray:
        """
        Re-scale feature based on fitted scaling processor

        :param feature_name: str
            Name of the feature

        :return: np.ndarray
            Re-scaled feature
        """
        return np.reshape(self.processing_memory['processor'][feature_name].transform(X=np.reshape(self.df[feature_name], (-1, 1))), (1, -1))[0]

    def robust_scaler(self,
                      feature_name: str,
                      with_centering: bool = True,
                      with_scaling: bool = True,
                      quantile_range: Tuple[float, float] = (0.25, 0.75),
                      ) -> np.ndarray:
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

        :return: np.ndarray
            Robust scaled feature
        """
        _robust: RobustScaler = RobustScaler(with_centering=with_centering, with_scaling=with_scaling, quantile_range=quantile_range)
        _robust.fit(np.reshape(self.df[feature_name], (-1, 1)), y=None)
        self.processor = _robust
        return np.reshape(_robust.transform(X=np.reshape(self.df[feature_name], (-1, 1))), (1, -1))[0]

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
                        ) -> np.ndarray:
        """
        Standardize feature

        :param feature_name: str
            Name of the feature to process

        :param with_mean: bool
            Using mean to standardize features

        :param with_std: bool
            Using standard deviation to standardize features

        :return: np.ndarray
            Standard scaled feature
        """
        _standard: StandardScaler = StandardScaler(with_mean=with_mean, with_std=with_std)
        _standard.fit(np.reshape(self.df[feature_name], (-1, 1)), y=None)
        self.processor = _standard
        return np.reshape(_standard.transform(X=np.reshape(self.df[feature_name], (-1, 1))), (1, -1))[0]

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
