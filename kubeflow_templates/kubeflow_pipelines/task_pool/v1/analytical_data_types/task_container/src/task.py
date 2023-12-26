"""

Task: ... (Function to run in container)

"""

import argparse
import pandas as pd

from analytical_data_types import AnalyticalDataTypes
from aws import save_file_to_s3
from file_handler import file_handler
from typing import Any, NamedTuple, Tuple


PARSER = argparse.ArgumentParser(description="receive analytical data types")
PARSER.add_argument('-data_set_path', type=str, required=True, default=None, help='file path of the data set')
PARSER.add_argument('-max_categories', type=int, required=False, default=50, help='maximum number of categories')
PARSER.add_argument('-date_edges', type=Any, required=False, default=None, help='date boundaries to identify datetime features')
PARSER.add_argument('-sep', type=str, required=False, default=',', help='column separator')
PARSER.add_argument('-output_file_path_analytical_data_types', type=str, required=True, default=None, help='file path of the analytical data types output')
PARSER.add_argument('-s3_output_file_path_analytical_data_types', type=str, required=False, default=None, help='S3 file path of the analytical data types output')
ARGS = PARSER.parse_args()


def analytical_data_types(data_set_path: str,
                          output_file_path_analytical_data_type: str,
                          max_categories: int = 50,
                          date_edges: Tuple[str, str] = None,
                          sep: str = ',',
                          s3_output_file_path_analytical_data_types: str = None,
                          ) -> NamedTuple('outputs', [('analytical_data_types', dict)]):
    """
    Evaluate analytical data types

    :param data_set_path: str
        Complete file path of the data set

    :param output_file_path_analytical_data_type: str
        Path of the analytical data type information to save

    :param max_categories: int
        Maximum number of categories for identifying feature as categorical

    :param date_edges: Tuple[str, str]
            Date boundaries to identify datetime features

    :param sep: str
        Separator

    :param s3_output_file_path_analytical_data_types: str
        Compelte file path of the analytical data types output

    :return: NamedTuple
        Analytical data types of given features
    """
    _df: pd.DataFrame = pd.read_csv(filepath_or_buffer=data_set_path, sep=sep)
    _analytical_data_types: AnalyticalDataTypes = AnalyticalDataTypes(df=_df,
                                                                      feature_names=_df.columns.tolist(),
                                                                      date_edges=date_edges,
                                                                      max_categories=max_categories
                                                                      )
    _analytical_data_type: dict = _analytical_data_types.main()
    file_handler(file_path=output_file_path_analytical_data_type, obj=_analytical_data_type)
    if s3_output_file_path_analytical_data_types is not None:
        save_file_to_s3(file_path=s3_output_file_path_analytical_data_types, obj=_analytical_data_type)
    return [_analytical_data_type]


if __name__ == '__main__':
    analytical_data_types(data_set_path=ARGS.data_set_path,
                          output_file_path_analytical_data_type=ARGS.output_file_path_analytical_data_type,
                          max_categories=ARGS.max_categories,
                          date_edges=ARGS.date_edges,
                          sep=ARGS.sep,
                          s3_output_file_path_analytical_data_types=ARGS.s3_output_file_path_analytical_data_types
                          )
