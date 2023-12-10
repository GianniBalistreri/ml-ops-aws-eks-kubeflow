"""

Task: ... (Function to run in container)

"""

import argparse
import json
import pandas as pd

from analytical_data_types import AnalyticalDataTypes
from aws import save_file_to_s3
from file_handler import file_handler
from typing import NamedTuple


PARSER = argparse.ArgumentParser(description="receive analytical data types")
PARSER.add_argument('-data_set_path', type=str, required=True, default=None, help='file path of the data set')
PARSER.add_argument('-output_file_path_analytical_data_type', type=str, required=True, default=None, help='file path of the analytical data type output')
PARSER.add_argument('-output_file_path_analytical_data_type_customized', type=str, required=False, default=None, help='complete customized file path of the analytical data type output')
PARSER.add_argument('-sep', type=str, required=False, default=',', help='column separator')
PARSER.add_argument('-max_categories', type=int, required=False, default=50, help='maximum number of categories')
ARGS = PARSER.parse_args()


def analytical_data_types(data_set_path: str,
                          output_file_path_analytical_data_type: str,
                          output_file_path_analytical_data_type_customized: str = None,
                          sep: str = ',',
                          max_categories: int = 50,
                          ) -> NamedTuple('outputs', [('analytical_data_types', dict)]):
    """
    Evaluate analytical data types

    :param data_set_path: str
        Complete file path of the data set

    :param output_file_path_analytical_data_type: str
        Path of the analytical data type information to save

    :param output_file_path_analytical_data_type_customized: str
        Complete customized output file paths

    :param sep: str
        Separator

    :param max_categories: int
        Maximum number of categories for identifying feature as categorical

    :return: NamedTuple
        Analytical data types of given features
    """
    _df: pd.DataFrame = pd.read_csv(filepath_or_buffer=data_set_path, sep=sep)
    _analytical_data_types: AnalyticalDataTypes = AnalyticalDataTypes(df=_df,
                                                                      feature_names=_df.columns.tolist(),
                                                                      date_edges=None,
                                                                      max_categories=max_categories
                                                                      )
    _analytical_data_type: dict = _analytical_data_types.main()
    file_handler(file_path=output_file_path_analytical_data_type, obj=_analytical_data_type)
    if output_file_path_analytical_data_type_customized is not None:
        save_file_to_s3(file_path=output_file_path_analytical_data_type_customized, obj=_analytical_data_type)
    return [_analytical_data_type]


if __name__ == '__main__':
    analytical_data_types(data_set_path=ARGS.data_set_path,
                          output_file_path_analytical_data_type=ARGS.output_file_path_analytical_data_type,
                          output_file_path_analytical_data_type_customized=ARGS.output_file_path_analytical_data_type_customized,
                          sep=ARGS.sep,
                          max_categories=ARGS.max_categories
                          )
