"""

Task: ... (Function to run in container)

"""

import argparse
import ast
import pandas as pd

from aws import load_file_from_s3, load_file_from_s3_as_df, save_file_to_s3_as_df
from custom_logger import Log
from file_handler import file_handler
from imputation import Imputation
from typing import Dict, List, NamedTuple, Union


PARSER = argparse.ArgumentParser(description="impute missing values")
PARSER.add_argument('-data_set_path', type=str, required=True, default=None, help='complete file path of the data set')
PARSER.add_argument('-features', nargs='+', required=False, default=None, help='names of the features to impute')
PARSER.add_argument('-imp_meth', type=str, required=False, default='multiple', help='imputation method')
PARSER.add_argument('-multiple_meth', type=str, required=False, default='random', help='multiple imputation method')
PARSER.add_argument('-single_meth', type=str, required=False, default='constant', help='single imputation method')
PARSER.add_argument('-constant_value', type=float, required=False, default=None, help='constant value (single imputation)')
PARSER.add_argument('-m', type=int, required=False, default=3, help='amount of chains to generate')
PARSER.add_argument('-convergence_threshold', type=float, required=False, default=0.99, help='convergence threshold (multiple imputation)')
PARSER.add_argument('-mice_config', type=str, required=False, default=None, help='config for using mice algorithm (multiple imputation)')
PARSER.add_argument('-imp_config', type=str, required=False, default=None, help='config for imputation for each feature')
PARSER.add_argument('-analytical_data_types_path', type=str, required=False, default=None, help='assignment of features to analytical data types')
PARSER.add_argument('-sep', type=str, required=False, default=',', help='column separator')
PARSER.add_argument('-output_file_path_imp_features', type=str, required=True, default=None, help='file path of the imputed feature names output')
PARSER.add_argument('-s3_output_path_imputed_data_set', type=str, required=True, default=None, help='S3 file path of the imputed data set output')
PARSER.add_argument('-kwargs', type=str, required=False, default=None, help='key-word arguments vor class Imputation')
ARGS = PARSER.parse_args()


def imputation(data_set_path: str,
               s3_output_path_imputed_data_set: str,
               output_file_path_imp_features: str,
               features: List[str] = None,
               imp_meth: str = 'multiple',
               multiple_meth: str = 'random',
               single_meth: str = 'constant',
               constant_value: Union[int, float] = None,
               m: int = 3,
               convergence_threshold: float = 0.99,
               mice_config: dict = None,
               imp_config: dict = None,
               analytical_data_types_path: str = None,
               sep: str = ',',
               **kwargs
               ) -> NamedTuple('outputs', [('features', list)]):
    """
    Impute missing values

    :param data_set_path: str
        Complete file path of the data set

    :param s3_output_path_imputed_data_set: str
        Complete file path of the imputed data set

    :param output_file_path_imp_features: str
        File path of the imputed feature names

    :param features: List[str]
        Name of the features

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

    :param analytical_data_types_path: str
        Complete file path of the analytical data types

    :param sep: str
        Separator

    :param kwargs: dict
        Key-word arguments for class Imputation

    :return: NamedTuple
        Names of the imputed features (not imputed features included)
    """
    if analytical_data_types_path is None:
        _analytical_data_types: Dict[str, List[str]] = {}
    else:
        _analytical_data_types: Dict[str, List[str]] = load_file_from_s3(file_path=analytical_data_types_path)
        Log().log(msg=f'Load analytical data types: {analytical_data_types_path}')
    _df: pd.DataFrame = load_file_from_s3_as_df(file_path=data_set_path, sep=sep)
    Log().log(msg=f'Load data set: {data_set_path} -> Cases={_df.shape[0]}, Features={_df.shape[1]}')
    if features is None:
        _features: List[str] = _df.columns.tolist()
    else:
        _features: List[str] = features
    _imputation: Imputation = Imputation(df=_df, **kwargs)
    if _imputation.missing_value_count == 0:
        _df_imp: pd.DataFrame = _df
    else:
        _df_imp: pd.DataFrame = pd.DataFrame(index=_df.index)
        if imp_config is None:
            _df_imp = _imputation.main(feature_names=_features,
                                       imp_meth=imp_meth,
                                       multiple_meth=multiple_meth,
                                       single_meth=single_meth,
                                       constant_value=constant_value,
                                       m=m,
                                       convergence_threshold=convergence_threshold,
                                       mice_config=mice_config
                                       )
        else:
            for feature in _features:
                if feature in imp_config.keys():
                    _imputed_feature = _imputation.main(feature_names=[feature],
                                                        imp_meth=imp_config[feature][0],
                                                        multiple_meth=imp_config[feature][1],
                                                        single_meth=imp_config[feature][1],
                                                        constant_value=imp_config[feature][2],
                                                        m=m,
                                                        convergence_threshold=convergence_threshold,
                                                        mice_config=mice_config
                                                        )
                    _df_imp[feature] = _imputed_feature
                else:
                    _df_imp[feature] = _df[feature]
    file_handler(file_path=output_file_path_imp_features, obj=_df_imp.columns.tolist())
    save_file_to_s3_as_df(file_path=s3_output_path_imputed_data_set, df=_df_imp, sep=sep)
    Log().log(msg=f'Save imputed data set: {s3_output_path_imputed_data_set}')
    return [_df_imp.columns.tolist()]


if __name__ == '__main__':
    if ARGS.features:
        ARGS.features = ast.literal_eval(ARGS.features[0])
    if ARGS.mice_config:
        ARGS.mice_config = ast.literal_eval(ARGS.mice_config)
    if ARGS.imp_config:
        ARGS.imp_config = ast.literal_eval(ARGS.imp_config)
    if ARGS.kwargs:
        ARGS.kwargs = ast.literal_eval(ARGS.kwargs)
    else:
        ARGS.kwargs = {}
    imputation(data_set_path=ARGS.data_set_path,
               features=ARGS.features,
               s3_output_path_imputed_data_set=ARGS.s3_output_path_imputed_data_set,
               output_file_path_imp_features=ARGS.output_file_path_imp_features,
               imp_meth=ARGS.imp_meth,
               multiple_meth=ARGS.multiple_meth,
               single_meth=ARGS.single_meth,
               constant_value=ARGS.constant_value,
               m=ARGS.m,
               convergence_threshold=ARGS.convergence_threshold,
               mice_config=ARGS.mice_config,
               imp_config=ARGS.imp_config,
               analytical_data_types_path=ARGS.analytical_data_types_path,
               sep=ARGS.sep,
               **ARGS.kwargs
               )
