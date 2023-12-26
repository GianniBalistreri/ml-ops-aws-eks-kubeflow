"""

Get analytical data type

"""

import numpy as np
import pandas as pd
import re

from custom_logger import Log
from typing import Dict, List, Tuple

ANALYTICAL_DATA_TYPES: List[str] = ['categorical',
                                    'ordinal',
                                    'continuous',
                                    'date',
                                    'text'
                                    ]

SPECIAL_CHARACTERS: List[str] = [' ', '^', '°', '!', '"', "'", '§', '$', '%', '&', '/', '(', ')', '=', '?', '`', '´',
                                 '<', '>', '|', '@', '€', '*', '+', '#', '-', '_', '.', ',', ':', ';'
                                 ]


class AnalyticalDataTypesException(Exception):
    """
    Class for handling exception for class AnalyticalDataTypes
    """
    pass


class AnalyticalDataTypes:
    """
    Class for receiving analytical data type from pandas DataFrame
    """
    def __init__(self,
                 df: pd.DataFrame,
                 feature_names: List[str],
                 date_edges: Tuple[str, str] = None,
                 max_categories: int = 100
                 ):
        """
        :param df: pd.DataFrame
            Data set

        :param feature_names: List[str]
            Name of the features to process

        :param date_edges: Tuple[str, str]
            Date boundaries to identify datetime features

        :param max_categories: int
            Maximum number of categories for identifying feature as categorical
        """
        self.df: pd.DataFrame = df
        self.feature_names: List[str] = feature_names
        self.date_edges: Tuple[str, str] = date_edges
        self.max_categories: int = max_categories

    def _get_analytical_data_type(self, feature_name: str) -> str:
        """
        Retrieve analytical data type from pandas DataFrame

        :param feature_name: str
            Name of the feature

        :return: str
            Name of the analytical data type
        """
        _feature_data = self.df.loc[~self.df[feature_name].isnull(), feature_name]
        _dtype: np.dtype = self.df[feature_name].dtype
        if str(_dtype).find('float') >= 0:
            _unique = _feature_data.unique()
            if any(_feature_data.isnull()):
                if any(_unique[~pd.isnull(_unique)] % 1) != 0:
                    Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "continuous" -> contains missing values, found float value')
                    return 'continuous'
                else:
                    if len(str(_feature_data.min()).split('.')[0]) > 4:
                        try:
                            assert pd.to_datetime(_feature_data)
                            if self.date_edges is None:
                                Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "date" -> contains missing values, no float value found, xxx, can be converted into datetime, no date edges were given')
                                return 'date'
                            else:
                                if (self.date_edges[0] < pd.to_datetime(_unique.min())) or (
                                        self.date_edges[1] > pd.to_datetime(_unique.max())):
                                    Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "id_text" -> contains missing values, no float value found, xxx, can be converted into datetime, out of date edges')
                                    return 'id_text'
                                else:
                                    Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "date" -> contains missing values, no float value found, xxx, can be converted into datetime, within given date edges')
                                    return 'date'
                        except (TypeError, ValueError):
                            Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "id_text" -> contains missing values, no float value found, xxx, cannot be converted into datetime')
                            return 'id_text'
                    else:
                        if len(_unique) > self.max_categories:
                            Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "ordinal" -> contains missing values, no float value found, xxx, more unique values than max category threshold')
                            return 'ordinal'
                        else:
                            Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "categorical" -> contains missing values, no float value found, xxx, less unique values than max category threshold')
                            return 'categorical'
            else:
                if any(_unique % 1) != 0:
                    Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "continuous" -> no missing values found, found float value')
                    return 'continuous'
                else:
                    if len(str(_feature_data.min()).split('.')[0]) > 4:
                        try:
                            assert pd.to_datetime(_feature_data)
                            if self.date_edges is None:
                                Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "date" -> no missing values found, no float value found, xxx, can be converted into datetime, no date edges were given')
                                return 'date'
                            else:
                                if (self.date_edges[0] < pd.to_datetime(_unique.min())) or (
                                        self.date_edges[1] > pd.to_datetime(_unique.max())):
                                    Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "id_text" -> no missing values found, no float value found, xxx, can be converted into datetime, out of date edges')
                                    return 'id_text'
                                else:
                                    Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "date" -> no missing values found, no float value found, xxx, can be converted into datetime, within given date edges')
                                    return 'date'
                        except (TypeError, ValueError):
                            Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "id_text" -> no missing values found, no float value found, xxx, cannot be converted into datetime')
                            return 'id_text'
                    else:
                        if len(_feature_data) == len(_feature_data.unique()):
                            Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "id_text" -> no missing values found, no float value found, xxx, contains unique values only')
                            return 'id_text'
                        if len(_unique) > self.max_categories:
                            Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "ordinal" -> no missing values found, no float value found, xxx, more unique values than max category threshold')
                            return 'ordinal'
                        else:
                            Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "categorical" -> no missing values found, no float value found, xxx, less unique values than max category threshold')
                            return 'categorical'
        elif str(_dtype).find('int') >= 0:
            if len(_feature_data) == len(_feature_data.unique()):
                Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "id_text" -> contains unique values only')
                return 'id_text'
            else:
                if len(_feature_data.unique()) > self.max_categories:
                    Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "ordinal" -> more unique values than max category threshold')
                    return 'ordinal'
                else:
                    Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "categorical" -> less unique values than max category threshold')
                    return 'categorical'
        elif str(_dtype).find('object') >= 0:
            _unique: np.array = _feature_data.unique()
            _digits: int = 0
            _dot: bool = False
            _max_dots: int = 0
            for text_val in _unique:
                if text_val == text_val:
                    if (str(text_val).find('.') >= 0) or (str(text_val).replace(',', '').isdigit()):
                        _dot = True
                    if str(text_val).replace('.', '').replace('-', '').isdigit() or str(text_val).replace(',', '').replace('-', '').isdigit():
                        if (len(str(text_val).split('.')) == 2) or (len(str(text_val).split(',')) == 2):
                            _digits += 1
                    if len(str(text_val).split('.')) > _max_dots:
                        _max_dots = len(str(text_val).split('.'))
            if _digits >= (len(_unique[~pd.isnull(_unique)]) * 0.5):
                if _dot:
                    try:
                        if any(_unique[~pd.isnull(_unique)] % 1) != 0:
                            Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "continuous" -> more than 50% are numeric values, found dot, found float values')
                            return 'continuous'
                        else:
                            if _max_dots == 2:
                                Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "continuous" -> more than 50% are numeric values, found dot, no float values found, max values around dots equal 2')
                                return 'continuous'
                            else:
                                Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "id_text" -> more than 50% are numeric values, found dot, no float values found, max values around dots unequal 2')
                                return 'id_text'
                    except (TypeError, ValueError):
                        if _max_dots == 2:
                            Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "continuous" -> more than 50% are numeric values, found dot, no float values found, max values around dots equal 2')
                            return 'continuous'
                        else:
                            Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "id_text" -> more than 50% are numeric values, found dot, no float values found, max values around dots unequal 2')
                            return 'id_text'
                else:
                    if len(_feature_data) == len(_feature_data.unique()):
                        Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "id_text" -> more than 50% are numeric values, no dot found, contains unique values only')
                        return 'id_text'
                    _len_of_feature = pd.DataFrame()
                    _len_of_feature[feature_name] = _feature_data[~_feature_data.isnull()]
                    _len_of_feature['len'] = _len_of_feature[feature_name].str.len()
                    _unique_values: np.array = _len_of_feature['len'].unique()
                    if len(_feature_data.unique()) >= (len(_feature_data) * 0.5):
                        Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "id_text" -> more than 50% are numeric values, no dot found, contains not only unique values, number of unique values greater than 50% of all values')
                        return 'id_text'
                    else:
                        if len(_feature_data.unique()) > self.max_categories:
                            Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "ordinal" -> more than 50% are numeric values, no dot found, contains not only unique values, number of unique values less than 50% of all values, more unique values than max category threshold')
                            return 'ordinal'
                        else:
                            Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "categorical" -> more than 50% are numeric values, no dot found, contains not only unique values, number of unique values less than 50% of all values, less unique values than max category threshold')
                            return 'categorical'
            else:
                try:
                    _potential_date = _feature_data[~_feature_data.isnull()]
                    _unique_years = pd.to_datetime(_potential_date).dt.year.unique()
                    _unique_months = pd.to_datetime(_potential_date).dt.isocalendar().week.unique()
                    _unique_days = pd.to_datetime(_potential_date).dt.day.unique()
                    _unique_cats: int = len(_unique_years) + len(_unique_months) + len(_unique_days)
                    if _unique_cats > 4:
                        Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "date" -> less than 50% are numeric values, unique date values are greater than 4')
                        return 'date'
                    else:
                        if len(_feature_data) == len(_feature_data.unique()):
                            Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "id_text" -> less than 50% are numeric values, unique date values are less than 4, contains unique values only')
                            return 'id_text'
                        if len(_feature_data.unique()) <= 3:
                            Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "categorical" -> less than 50% are numeric values, unique date values are less than 4, contains not only unique values, unique values less than 4')
                            return 'categorical'
                        else:
                            _len_of_feature = pd.DataFrame()
                            _len_of_feature[feature_name] = _feature_data[~_feature_data.isnull()]
                            _len_of_feature['len'] = _len_of_feature[feature_name].str.len()
                            _unique_values: np.array = _len_of_feature['len'].unique()
                            for val in _unique_values:
                                if len(re.findall(pattern=r'[a-zA-Z]', string=str(val))) > 0:
                                    if len(_feature_data.unique()) >= (len(_feature_data) * 0.5):
                                        Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "id_text" -> less than 50% are numeric values, unique date values are less than 4, contains not only unique values, unique values greater than 3, found characters, number of unique values greater than 50% of all values')
                                        return 'id_text'
                                    else:
                                        Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "categorical" -> less than 50% are numeric values, unique date values are less than 4, contains not only unique values, unique values greater than 3, found characters, number of unique values less than 50% of all values')
                                        return 'categorical'
                            if np.min(_unique_values) > 3:
                                if len(_feature_data.unique()) >= (len(_feature_data) * 0.5):
                                    Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "id_text" -> less than 50% are numeric values, unique date values are less than 4, contains not only unique values, unique values greater than 3, no characters found, minimum values greater than 3, number of unique values greater than 50% of all values')
                                    return 'id_text'
                                else:
                                    if len(_feature_data.unique()) > self.max_categories:
                                        Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "ordinal" -> less than 50% are numeric values, unique date values are less than 4, contains not only unique values, unique values greater than 3, no characters found, minimum values greater than 3, number of unique values less than 50% of all values, more unique values than max category threshold')
                                        return 'ordinal'
                                    else:
                                        Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "categorical" -> less than 50% are numeric values, unique date values are less than 4, contains not only unique values, unique values greater than 3, no characters found, minimum values greater than 3, number of unique values less than 50% of all values, less unique values than max category threshold')
                                        return 'categorical'
                            else:
                                Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "categorical" -> less than 50% are numeric values, unique date values are less than 4, contains not only unique values, unique values greater than 3, no characters found, minimum values less than 4')
                                return 'categorical'
                except (TypeError, ValueError):
                    if len(_feature_data) == len(_unique):
                        Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "id_text" -> less than 50% are numeric values, contains unique values only')
                        return 'id_text'
                    if len(_feature_data.unique()) <= 3:
                        Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "categorical" -> less than 50% are numeric values, contains not only unique values, unique values less than 4')
                        return 'categorical'
                    else:
                        _len_of_feature = _feature_data[~_feature_data.isnull()]
                        _len_of_feature['len'] = _len_of_feature.str.len()
                        _unique_values: np.array = _len_of_feature['len'].unique()
                        for val in _feature_data.unique():
                            if len(re.findall(pattern=r'[a-zA-Z]', string=str(val))) > 0:
                                if len(_feature_data.unique()) >= (len(_feature_data) * 0.5):
                                    Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "id_text" -> less than 50% are numeric values, unique date values are less than 4, contains not only unique values, unique values greater than 3, found characters, number of unique values greater than 50% of all values')
                                    return 'id_text'
                                else:
                                    if len(_feature_data.unique()) > self.max_categories:
                                        Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "ordinal" -> less than 50% are numeric values, unique date values are less than 4, contains not only unique values, unique values greater than 3, no characters found, number of unique values greater than 50% of all values, more unique values than max category threshold')
                                        return 'ordinal'
                                    else:
                                        Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "categorical" -> less than 50% are numeric values, unique date values are less than 4, contains not only unique values, unique values greater than 3, no characters found, number of unique values greater than 50% of all values, less unique values than max category threshold')
                                        return 'categorical'
                        for ch in SPECIAL_CHARACTERS:
                            if any(_len_of_feature.str.find(ch) > 0):
                                if len(_feature_data.unique()) >= (len(_feature_data) * 0.5):
                                    Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "id_text" -> less than 50% are numeric values, contains not only unique values, unique values greater than 3, found special characters, number of unique values greater than 50% of all values')
                                    return 'id_text'
                                else:
                                    Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "categorical" -> less than 50% are numeric values, contains not only unique values, unique values greater than 3, found special characters, number of unique values less than 50% of all values')
                                    return 'categorical'
                        # if np.mean(_unique_values) == np.median(_unique_values):
                        #    return 'id_text'
                        # else:
                        Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "categorical" -> less than 50% are numeric values, contains not only unique values, unique values greater than 3, no characters found, no special characters found')
                        return 'categorical'
        elif str(_dtype).find('date') >= 0:
            Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "date" -> dtype date')
            return 'date'
        elif str(_dtype).find('bool') >= 0:
            Log().log(msg=f'Feature "{feature_name}" has dtype "{_dtype}" and analytical type "categorical" -> dtype bool')
            return 'categorical'

    def main(self) -> Dict[str, str]:
        """
        Run retrieving analytical data type

        :return: Dict[str, str]
            Analytical data type for each feature
        """
        _analytical_data_types: dict = {}
        for feature in self.feature_names:
            _analytical_data_types.update({feature: self._get_analytical_data_type(feature_name=feature)})
        return _analytical_data_types
