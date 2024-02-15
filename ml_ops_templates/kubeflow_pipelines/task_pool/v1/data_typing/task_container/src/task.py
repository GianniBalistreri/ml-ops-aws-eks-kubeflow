"""

Task: ... (Function to run in container)

"""

import argparse
import ast
import pandas as pd

from aws import load_file_from_s3, load_file_from_s3_as_df, save_file_to_s3, save_file_to_s3_as_df
from custom_logger import Log
from data_typing import DataTyping
from typing import Dict, List


PARSER = argparse.ArgumentParser(description="convert data types")
PARSER.add_argument('-data_set_path', type=str, required=True, default=None, help='file path of the data set')
PARSER.add_argument('-analytical_data_types_path', type=str, required=True, default=None, help='assignment of features to analytical data types')
PARSER.add_argument('-missing_value_features', nargs='+', required=False, default=None, help='feature names containing missing values')
PARSER.add_argument('-data_types_config', type=str, required=False, default=None, help='pre-defined data typing configuration')
PARSER.add_argument('-sep', type=str, required=False, default=',', help='column separator')
PARSER.add_argument('-s3_output_file_path_data_set', type=str, required=True, default=None, help='S3 file path of the typed data set output')
PARSER.add_argument('-s3_output_file_path_data_typing', type=str, required=False, default=None, help='S3 file path of the data typing output')
ARGS = PARSER.parse_args()


def data_typing(data_set_path: str,
                analytical_data_types_path: str,
                s3_output_file_path_data_set: str,
                missing_value_features: List[str] = None,
                data_types_config: Dict[str, str] = None,
                sep: str = ',',
                s3_output_file_path_data_typing: str = None
                ) -> None:
    """
    Type features of structured (tabular) data

    :param data_set_path: str
        Complete file path of the data set

    :param analytical_data_types_path: str
        Complete file path of the analytical data types

    :param s3_output_file_path_data_set: str
        Complete file path of the typed data set

    :param missing_value_features: List[str]
        Name of the features containing missing values

    :param data_types_config: Dict[str, str]
        Pre-defined data typing configuration

    :param sep: str
        Separator

    :param s3_output_file_path_data_typing: str
        Complete file path of the data typing output
    """
    _analytical_data_types: Dict[str, List[str]] = load_file_from_s3(file_path=analytical_data_types_path)
    Log().log(msg=f'Load analytical data types: {analytical_data_types_path}')
    _df: pd.DataFrame = load_file_from_s3_as_df(file_path=data_set_path, sep=sep)
    Log().log(msg=f'Load data set: {data_set_path} -> Cases={_df.shape[0]}, Features={_df.shape[1]}')
    _feature_names: List[str] = _df.columns.tolist()
    _data_typing: DataTyping = DataTyping(df=_df,
                                          feature_names=_feature_names,
                                          analytical_data_types=_analytical_data_types,
                                          missing_value_features=missing_value_features,
                                          data_types_config=data_types_config
                                          )
    _data_typing.main()
    save_file_to_s3_as_df(file_path=s3_output_file_path_data_set, df=_data_typing.df, sep=sep)
    Log().log(msg=f'Save typed data set: {s3_output_file_path_data_set}')
    if s3_output_file_path_data_typing is not None:
        save_file_to_s3(file_path=s3_output_file_path_data_typing, obj=_data_typing.data_types_config)
        Log().log(msg=f'Save data typing: {s3_output_file_path_data_typing}')


if __name__ == '__main__':
    if ARGS.data_types_config:
        ARGS.data_types_config = ast.literal_eval(ARGS.data_types_config)
    data_typing(data_set_path=ARGS.data_set_path,
                analytical_data_types_path=ARGS.analytical_data_types_path,
                s3_output_file_path_data_set=ARGS.s3_output_file_path_data_set,
                missing_value_features=ARGS.missing_value_features,
                data_types_config=ARGS.data_types_config,
                sep=ARGS.sep,
                s3_output_file_path_data_typing=ARGS.s3_output_file_path_data_typing
                )
