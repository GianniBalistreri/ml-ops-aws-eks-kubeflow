"""

Task: ... (Function to run in container)

"""

import argparse
import pandas as pd

from imputation import Imputation
from kfp.v2.dsl import OutputPath
from typing import Any, NamedTuple, List, Union


PARSER = argparse.ArgumentParser(description="impute missing values")
PARSER.add_argument('-data_set_path', type=str, required=True, default=None, help='complete file path of the data set')
PARSER.add_argument('-feature_names', type=Any, required=True, default=None, help='names of the features to impute')
PARSER.add_argument('-prediction_feature_name', type=str, required=True, default=None, help='name of the prediction variable')
PARSER.add_argument('-imp_meth', type=str, required=False, default='multiple', help='imputation method')
PARSER.add_argument('-multiple_meth', type=str, required=False, default='random', help='multiple imputation method')
PARSER.add_argument('-single_meth', type=str, required=False, default='constant', help='single imputation method')
PARSER.add_argument('-constant_value', type=Any, required=False, default=None, help='constant value (single imputation)')
PARSER.add_argument('-m', type=int, required=False, default=3, help='amount of chains to generate')
PARSER.add_argument('-convergence_threshold', type=float, required=False, default=0.99, help='convergence threshold (multiple imputation)')
PARSER.add_argument('-mice_config', type=Any, required=False, default=None, help='config for using mice algorithm (multiple imputation)')
PARSER.add_argument('-imp_config', type=Any, required=False, default=None, help='config for imputation for each feature')
PARSER.add_argument('-sep', type=str, required=False, default=',', help='column separator')
PARSER.add_argument('-output_bucket_name', type=str, required=False, default=None, help='output S3 bucket name')
ARGS = PARSER.parse_args()


def imputation(data_set_path: str,
               feature_names: List[str],
               output_file_path: str,
               imp_meth: str = 'multiple',
               multiple_meth: str = 'random',
               single_meth: str = 'constant',
               constant_value: Union[int, float] = None,
               m: int = 3,
               convergence_threshold: float = 0.99,
               mice_config: dict = None,
               imp_config: dict = None,
               sep: str = ',',
               ) -> NamedTuple('outputs', [('imputed_data_set_path', str)]):
    """
    Impute missing values

    :param data_set_path: str
        Complete file path of the data set

    :param feature_names: List[str]
        Name of the features

    :param output_file_path: str
        Complete path of the imputed data set

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

    :param constant_value: Union[int, float]
        Constant imputation value used for single imputation method constant

    :param m: int
        Number of chains (multiple imputation)

    :param convergence_threshold: float
        Convergence threshold used for multiple imputation

    :param mice_config: dict
        bla

    :param imp_config: dict
        Assignment of different imputation methods to features
            -> key: feature name
            -> value: tuple(imp_meth, meth, constant_value)

    :param sep: str
        Separator

    :return: NamedTuple
        Complete file path of the imputed data set
    """
    _df: pd.DataFrame = pd.read_csv(filepath_or_buffer=data_set_path, sep=sep)
    _imputation: Imputation = Imputation(df=_df)
    if imp_config is None:
        _df_imp: pd.DataFrame = _imputation.main(feature_names=feature_names,
                                                 imp_meth=imp_meth,
                                                 multiple_meth=multiple_meth,
                                                 single_meth=single_meth,
                                                 constant_value=constant_value,
                                                 m=m,
                                                 convergence_threshold=convergence_threshold,
                                                 mice_config=mice_config
                                                 )
    else:
        _df_imp: pd.DataFrame = pd.DataFrame()
        for feature in imp_config.keys():
            _df_imp = pd.concat(objs=[_df_imp, _imputation.main(feature_names=[feature],
                                                                imp_meth=imp_config[feature][0],
                                                                multiple_meth=imp_config[feature][1],
                                                                single_meth=imp_config[feature][1],
                                                                constant_value=imp_config[feature][2],
                                                                m=m,
                                                                convergence_threshold=convergence_threshold,
                                                                mice_config=mice_config
                                                                )
                                      ]
                                )
    _df_imp.to_csv(path_or_buf=output_file_path, sep=sep, index=False)
    return [output_file_path]
