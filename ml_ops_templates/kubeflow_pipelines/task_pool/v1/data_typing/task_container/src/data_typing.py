"""

Data typing

"""

import pandas as pd

from custom_logger import Log
from typing import Dict, List


class DataTypingException(Exception):
    """
    Class for handling exception for class DataTyping
    """
    pass


class DataTyping:
    """
    Class for data typing
    """
    def __init__(self,
                 df: pd.DataFrame,
                 feature_names: List[str],
                 analytical_data_types: Dict[str, List[str]],
                 missing_value_features: List[str] = None,
                 data_types_config: Dict[str, str] = None
                 ):
        """
        :param df: pd.DataFrame
            Data set

        :param feature_names: str
            Name of the features to analyse

        :param missing_value_features: List[str]
            Name of the features containing missing values

        :param data_types_config: Dict[str, str]
            Pre-defined data typing configuration
        """
        self.df: pd.DataFrame = df
        self.feature: str = None
        self.feature_names: List[str] = feature_names
        self.analytical_data_types: Dict[str, List[str]] = analytical_data_types
        self.missing_value_features: List[str] = [] if missing_value_features is None else missing_value_features
        self.data_types_config: Dict[str, str] = {} if data_types_config is None else data_types_config

    def _process_boolean(self) -> None:
        """
        Process boolean typed feature
        """
        self.data_types_config.update({self.feature: 'int'})
        self.df[self.feature] = self.df[self.feature].astype(dtype=int)
        Log().log(msg=f'Convert type of feature "{self.feature}" from boolean to integer')

    def _process_date(self) -> None:
        """
        Process datetime typed feature
        """
        if self.feature in self.analytical_data_types['categorical']:
            if self.feature in self.missing_value_features:
                Log().log(msg=f'Cannot convert type of feature "{self.feature}" from date to integer because it contains missing values')
            else:
                self.data_types_config.update({self.feature: 'int'})
                self.df[self.feature] = self.df[self.feature].astype(dtype=int)
                Log().log(msg=f'Convert type of feature "{self.feature}" from date to integer')
        elif self.feature in self.analytical_data_types['continuous']:
            self.data_types_config.update({self.feature: 'float'})
            self.df[self.feature] = self.df[self.feature].astype(dtype=float)
            Log().log(msg=f'Convert type of feature "{self.feature}" from date to float')
        elif self.feature in self.analytical_data_types['id_text']:
            self.data_types_config.update({self.feature: 'str'})
            self.df[self.feature] = self.df[self.feature].astype(dtype=str)
            Log().log(msg=f'Convert type of feature "{self.feature}" from date to string')

    def _process_float(self) -> None:
        """
        Process float typed feature
        """
        if self.feature in self.analytical_data_types['categorical']:
            if self.feature in self.missing_value_features:
                Log().log(msg=f'Cannot convert type of feature "{self.feature}" from float to integer because it contains missing values')
            else:
                self.data_types_config.update({self.feature: 'int'})
                self.df[self.feature] = self.df[self.feature].astype(dtype=int)
                Log().log(msg=f'Convert type of feature "{self.feature}" from float to integer')
        elif self.feature in self.analytical_data_types['date']:
            self.data_types_config.update({self.feature: 'date'})
            self.df[self.feature] = pd.to_datetime(self.df[self.feature].values, errors='coerce')
            Log().log(msg=f'Convert type of feature "{self.feature}" from float to date')
        elif self.feature in self.analytical_data_types['id_text']:
            self.data_types_config.update({self.feature: 'str'})
            self.df[self.feature] = self.df[self.feature].astype(dtype=str)
            Log().log(msg=f'Convert type of feature "{self.feature}" from float to string')

    def _process_string(self) -> None:
        """
        Process string typed feature
        """
        if self.feature in self.analytical_data_types['continuous']:
            self.data_types_config.update({self.feature: 'float'})
            self.df[self.feature] = self.df[self.feature].astype(float)
            Log().log(msg=f'Convert type of feature "{self.feature}" from string to float')
        elif self.feature in self.analytical_data_types['categorical']:
            if any(self.df[self.feature].str.findall(pat='[a-z,A-Z]').isnull()):
                if self.feature in self.missing_value_features:
                    Log().log(msg=f'Cannot convert type of feature "{self.feature}" from string to integer because it contains missing values')
                else:
                    self.data_types_config.update({self.feature: 'int'})
                    self.df[self.feature] = self.df[self.feature].astype(int)
                    Log().log(msg=f'Convert type of feature "{self.feature}" from string to integer')
            else:
                Log().log(msg=f'Cannot convert type of feature "{self.feature}" from string to integer because it contains text values')
        elif self.feature in self.analytical_data_types['date']:
            self.data_types_config.update({self.feature: 'date'})
            self.df[self.feature] = pd.to_datetime(self.df[self.feature].values, errors='coerce')
            Log().log(msg=f'Convert type of feature "{self.feature}" from string to date')

    def main(self) -> None:
        """
        Run data typing
        """
        for feature in self.feature_names:
            if feature not in self.df.columns:
                Log().log(msg=f'Feature "{feature}" could not be found in data set')
                continue
            if feature in self.data_types_config.keys():
                if self.data_types_config.get(feature).find('float') >= 0:
                    self.df[feature] = self.df[feature].astype(dtype=float)
                elif self.data_types_config.get(feature).find('int') >= 0:
                    self.df[feature] = self.df[feature].astype(dtype=int)
                elif self.data_types_config.get(feature).find('str') >= 0:
                    self.df[feature] = self.df[feature].astype(dtype=str)
                elif self.data_types_config.get(feature).find('date') >= 0:
                    self.df[feature] = pd.to_datetime(self.df[feature].values, errors='coerce')
                elif self.data_types_config.get(feature).find('[ns]') >= 0:
                    self.df[feature] = pd.to_datetime(self.df[feature].values, errors='coerce')
                else:
                    raise DataTypingException(f'Type format ({self.data_types_config.get(feature)}) not supported')
                continue
            self.feature = feature
            try:
                if str(self.df[feature].dtype).find('object') >= 0:
                    self._process_string()
                elif str(self.df[feature].dtype).find('[ns]') >= 0:
                    self._process_date()
                elif str(self.df[feature].dtype).find('float') >= 0:
                    self._process_float()
                elif str(self.df[feature].dtype).find('bool') >= 0:
                    self._process_boolean()
                else:
                    Log().log(msg=f'No need to convert feature ({self.feature}) of type ({self.df[feature].dtype})')
            except (AttributeError, ValueError, TypeError) as e:
                Log().log(msg=f'Feature "{feature}" of type {str(self.df[self.feature].dtype)} could not be converted\nError: {e}')
