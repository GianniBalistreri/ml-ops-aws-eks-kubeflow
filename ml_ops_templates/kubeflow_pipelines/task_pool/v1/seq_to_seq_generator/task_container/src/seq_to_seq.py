"""

Text classification generator: generates transformer based neural networks

"""

import numpy as np
import pandas as pd

from custom_logger import Log
from typing import List


class TransformerGeneratorException(Exception):
    """
    Class for handling exception for class DataHealthCheck
    """
    pass


class TransformerGenerator:
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

