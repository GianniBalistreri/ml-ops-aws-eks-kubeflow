"""

Task: ... (Function to run in container)

"""

import argparse
import ast
import pandas as pd

from analytical_data_types import AnalyticalDataTypes
from aws import load_file_from_s3_as_df, save_file_to_s3
from custom_logger import Log
from file_handler import file_handler
from resource_metrics import get_available_cpu, get_cpu_utilization, get_cpu_utilization_per_core, get_memory, get_memory_utilization
from typing import Dict, List, NamedTuple, Tuple


PARSER = argparse.ArgumentParser(description="receive analytical data types")
PARSER.add_argument('-data_set_path', type=str, required=True, default=None, help='file path of the data set')
PARSER.add_argument('-max_categories', type=int, required=False, default=50, help='maximum number of categories')
PARSER.add_argument('-date_edges', nargs='+', type=int, required=False, default=None, help='date boundaries to identify datetime features')
PARSER.add_argument('-categorical', nargs='+', required=False, default=None, help='pre-assigned categorical features')
PARSER.add_argument('-ordinal', nargs='+', required=False, default=None, help='pre-assigned ordinal features')
PARSER.add_argument('-continuous', nargs='+', required=False, default=None, help='pre-assigned continuous features')
PARSER.add_argument('-date', nargs='+', required=False, default=None, help='pre-assigned categorical datetime features')
PARSER.add_argument('-id_text', nargs='+', required=False, default=None, help='pre-assigned categorical id / text features')
PARSER.add_argument('-sep', type=str, required=False, default=',', help='column separator')
PARSER.add_argument('-parallel_mode', type=int, required=False, default=0, help='whether to run task in parallel mode or not')
PARSER.add_argument('-output_file_path_analytical_data_types', type=str, required=True, default=None, help='file path of the analytical data types output')
PARSER.add_argument('-output_file_path_categorical_features', type=str, required=True, default=None, help='file path of the categorical features output')
PARSER.add_argument('-output_file_path_ordinal_features', type=str, required=True, default=None, help='file path of the ordinal features output')
PARSER.add_argument('-output_file_path_continuous_features', type=str, required=True, default=None, help='file path of the continuous features output')
PARSER.add_argument('-output_file_path_date_features', type=str, required=True, default=None, help='file path of the date features output')
PARSER.add_argument('-output_file_path_id_text_features', type=str, required=True, default=None, help='file path of the id / text features output')
PARSER.add_argument('-s3_output_file_path_analytical_data_types', type=str, required=True, default=None, help='S3 file path of the analytical data types output')
ARGS = PARSER.parse_args()


def analytical_data_types(data_set_path: str,
                          output_file_path_analytical_data_types: str,
                          output_file_path_categorical_features: str,
                          output_file_path_ordinal_features: str,
                          output_file_path_continuous_features: str,
                          output_file_path_date_features: str,
                          output_file_path_id_text_features: str,
                          s3_output_file_path_analytical_data_types: str,
                          max_categories: int = 50,
                          date_edges: Tuple[str, str] = None,
                          categorical: List[str] = None,
                          ordinal: List[str] = None,
                          continuous: List[str] = None,
                          date: List[str] = None,
                          id_text: List[str] = None,
                          sep: str = ',',
                          parallel_mode: bool = False,
                          ) -> NamedTuple('outputs', [('analytical_data_types', dict),
                                                      ('categorical_features', list),
                                                      ('ordinal_features', list),
                                                      ('continuous_features', list),
                                                      ('date_features', list),
                                                      ('id_text_features', list)
                                                      ]
                                          ):
    """
    Receive analytical data types

    :param data_set_path: str
        Complete file path of the data set

    :param output_file_path_analytical_data_types: str
        File path of the analytical data types output

    :param output_file_path_categorical_features: str
        File path of the categorical features output

    :param output_file_path_ordinal_features: str
        File path of the ordinal features output

    :param output_file_path_continuous_features: str
        File path of the continuous features output

    :param output_file_path_date_features: str
        File path of the date features output

    :param output_file_path_id_text_features: str
        File path of the id / text features output

    :param s3_output_file_path_analytical_data_types: str
        Complete file path of the analytical data types output

    :param max_categories: int
        Maximum number of categories for identifying feature as categorical

    :param date_edges: Tuple[str, str]
            Date boundaries to identify datetime features

    :param categorical: List[str]
            Pre-assigned categorical features

    :param ordinal: List[str]
        Pre-assigned ordinal features

    :param continuous: List[str]
        Pre-assigned continuous features

    :param date: List[str]
        Pre-assigned date features

    :param id_text: List[str]
        Pre-assigned id_text features

    :param parallel_mode: bool
        Whether to run task in parallel mode or not

    :param sep: str
        Separator

    :return: NamedTuple
        Analytical data types of given features
    """
    _cpu_available: int = get_available_cpu(logging=True)
    _memory_total: float = get_memory(total=True, logging=True)
    _memory_available: float = get_memory(total=False, logging=True)
    _df: pd.DataFrame = load_file_from_s3_as_df(file_path=data_set_path, sep=sep)
    Log().log(msg=f'Load data set: {data_set_path} -> Cases={_df.shape[0]}, Features={_df.shape[1]}')
    _analytical_data_types: AnalyticalDataTypes = AnalyticalDataTypes(df=_df,
                                                                      feature_names=_df.columns.tolist(),
                                                                      date_edges=date_edges,
                                                                      max_categories=max_categories
                                                                      )
    _analytical_data_type: Dict[str, List[str]] = _analytical_data_types.main(categorical=categorical,
                                                                              ordinal=ordinal,
                                                                              continuous=continuous,
                                                                              date=date,
                                                                              id_text=id_text
                                                                              )
    _s3_output_file_path_analytical_data_types: str = s3_output_file_path_analytical_data_types
    if parallel_mode:
        _suffix: str = data_set_path.split('.')[0].split('_')[-1]
        _s3_output_file_path_analytical_data_types = s3_output_file_path_analytical_data_types.replace('.', f'_{_suffix}.')
    save_file_to_s3(file_path=s3_output_file_path_analytical_data_types, obj=_analytical_data_type)
    Log().log(msg=f'Save analytical data types: {_s3_output_file_path_analytical_data_types}')
    file_handler(file_path=output_file_path_analytical_data_types, obj=_analytical_data_type)
    file_handler(file_path=output_file_path_categorical_features, obj=_analytical_data_type.get('categorical'))
    file_handler(file_path=output_file_path_ordinal_features, obj=_analytical_data_type.get('ordinal'))
    file_handler(file_path=output_file_path_continuous_features, obj=_analytical_data_type.get('continuous'))
    file_handler(file_path=output_file_path_date_features, obj=_analytical_data_type.get('date'))
    file_handler(file_path=output_file_path_id_text_features, obj=_analytical_data_type.get('id_text'))
    _cpu_utilization: float = get_cpu_utilization(interval=1, logging=True)
    _cpu_utilization_per_cpu: List[float] = get_cpu_utilization_per_core(interval=1, logging=True)
    _memory_utilization: float = get_memory_utilization(logging=True)
    _memory_available = get_memory(total=False, logging=True)
    return [_analytical_data_type,
            _analytical_data_type.get('categorical'),
            _analytical_data_type.get('ordinal'),
            _analytical_data_type.get('continuous'),
            _analytical_data_type.get('date'),
            _analytical_data_type.get('id_text')
            ]


if __name__ == '__main__':
    if ARGS.date_edges:
        ARGS.date_edges = tuple(ARGS.date_edges)
    if ARGS.categorical:
        ARGS.categorical = ast.literal_eval(ARGS.categorical[0])
    if ARGS.ordinal:
        ARGS.ordinal = ast.literal_eval(ARGS.ordinal[0])
    if ARGS.continuous:
        ARGS.continuous = ast.literal_eval(ARGS.continuous[0])
    if ARGS.date:
        ARGS.date = ast.literal_eval(ARGS.date[0])
    if ARGS.id_text:
        ARGS.id_text = ast.literal_eval(ARGS.id_text[0])
    analytical_data_types(data_set_path=ARGS.data_set_path,
                          output_file_path_analytical_data_types=ARGS.output_file_path_analytical_data_types,
                          output_file_path_categorical_features=ARGS.output_file_path_categorical_features,
                          output_file_path_ordinal_features=ARGS.output_file_path_ordinal_features,
                          output_file_path_continuous_features=ARGS.output_file_path_continuous_features,
                          output_file_path_date_features=ARGS.output_file_path_date_features,
                          output_file_path_id_text_features=ARGS.output_file_path_id_text_features,
                          s3_output_file_path_analytical_data_types=ARGS.s3_output_file_path_analytical_data_types,
                          max_categories=ARGS.max_categories,
                          date_edges=ARGS.date_edges,
                          categorical=ARGS.categorical,
                          ordinal=ARGS.ordinal,
                          continuous=ARGS.continuous,
                          date=ARGS.date,
                          id_text=ARGS.id_text,
                          sep=ARGS.sep,
                          parallel_mode=ARGS.parallel_mode
                          )
