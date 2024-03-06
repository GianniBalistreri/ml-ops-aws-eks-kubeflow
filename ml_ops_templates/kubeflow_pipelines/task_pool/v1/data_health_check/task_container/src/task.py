"""

Task: ... (Function to run in container)

"""

import argparse
import ast
import pandas as pd

from aws import load_file_from_s3, load_file_from_s3_as_df, save_file_to_s3
from custom_logger import Log
from data_health_check import DataHealthCheck
from file_handler import file_handler
from resource_metrics import get_available_cpu, get_cpu_utilization, get_cpu_utilization_per_core, get_memory, get_memory_utilization
from typing import List, NamedTuple


PARSER = argparse.ArgumentParser(description="check data health")
PARSER.add_argument('-data_set_path', type=str, required=True, default=None, help='file path of the data set')
PARSER.add_argument('-analytical_data_types_path', type=str, required=True, default=None, help='assignment of features to analytical data types')
PARSER.add_argument('-features', nargs='+', required=False, default=None, help='feature names used for data health check')
PARSER.add_argument('-missing_value_threshold', type=float, required=False, default=0.95, help='threshold to classify features as invalid based on the amount of missing values')
PARSER.add_argument('-parallel_mode', type=int, required=False, default=0, help='whether to run task in parallel mode or not')
PARSER.add_argument('-sep', type=str, required=False, default=',', help='column separator')
PARSER.add_argument('-output_file_path_missing_data', type=str, required=True, default=None, help='file path of features containing too much missing data output')
PARSER.add_argument('-output_file_path_invariant_features', type=str, required=True, default=None, help='file path of invariant features output')
PARSER.add_argument('-output_file_path_duplicated_features', type=str, required=True, default=None, help='file path of duplicated features output')
PARSER.add_argument('-output_file_path_valid_features', type=str, required=True, default=None, help='file path of valid features output')
PARSER.add_argument('-output_file_path_prop_valid_features', type=str, required=True, default=None, help='file path of the proportion of valid features output')
PARSER.add_argument('-output_file_path_n_valid_features', type=str, required=True, default=None, help='file path of number of valid features output')
PARSER.add_argument('-s3_output_file_path_data_health_check', type=str, required=False, default=None, help='S3 file path of the data health check output')
ARGS = PARSER.parse_args()


def data_health_check(data_set_path: str,
                      analytical_data_types_path: str,
                      output_file_path_missing_data: str,
                      output_file_path_invariant_features: str,
                      output_file_path_duplicated_features: str,
                      output_file_path_valid_features: str,
                      output_file_path_prop_valid_features: str,
                      output_file_path_n_valid_features: str,
                      features: List[str] = None,
                      missing_value_threshold: float = 0.95,
                      parallel_mode: bool = False,
                      sep: str = ',',
                      s3_output_file_path_data_health_check: str = None
                      ) -> NamedTuple('outputs', [('missing_data', list),
                                                  ('valid_features', list),
                                                  ('prop_valid_features', float),
                                                  ('n_valid_features', int)
                                                  ]
                                      ):
    """
    Examine data health of structured (tabular) data

    :param data_set_path: str
        Complete file path of the data set

    :param analytical_data_types_path: str
        Complete file path of the analytical data types

    :param output_file_path_missing_data: str
        Path of the features containing too much missing data output

    :param output_file_path_invariant_features: str
        Path of the invariant features

    :param output_file_path_duplicated_features: str
        Path of the duplicated features

    :param output_file_path_valid_features: str
        Path of the valid features output

    :param output_file_path_prop_valid_features: str
        Path of the proportion of valid features output

    :param output_file_path_n_valid_features: str
        Path of the number of valid features output

    :param features: List[str]
        Name of the features to check

    :param missing_value_threshold: float
        Threshold of missing values to exclude numeric feature

    :param parallel_mode: bool
        Whether to run task in parallel mode or not

    :param sep: str
        Separator

    :param s3_output_file_path_data_health_check: str
        Complete file path of the data health check output

    :return: NamedTuple
        Data health check of given features, invalid and valid features, proportion of valid features
    """
    _cpu_available: int = get_available_cpu(logging=True)
    _memory_total: float = get_memory(total=True, logging=True)
    _memory_available: float = get_memory(total=False, logging=True)
    _analytical_data_types: dict = load_file_from_s3(file_path=analytical_data_types_path)
    Log().log(msg=f'Load analytical data types: {analytical_data_types_path}')
    _df: pd.DataFrame = load_file_from_s3_as_df(file_path=data_set_path, sep=sep)
    Log().log(msg=f'Load data set: {data_set_path} -> Cases={_df.shape[0]}, Features={_df.shape[1]}')
    _features_in_data_set: List[str] = _df.columns.tolist()
    _features: List[str] = []
    if features is None:
        _features = _features_in_data_set
    else:
        for feature in features:
            if feature in _features_in_data_set:
                _features.append(feature)
    _data_health_check: DataHealthCheck = DataHealthCheck(df=_df, feature_names=_features)
    _data_health_check_results: dict = _data_health_check.main()
    _relevant_features_to_check_missing_values: List[str] = _analytical_data_types.get('continuous')
    _relevant_features_to_check_missing_values.extend(_analytical_data_types.get('date'))
    _valid_features: List[str] = []
    _invariant_features: List[str] = []
    _duplicated_features: List[str] = []
    _numeric_features_containing_missing_data: List[str] = []
    for feature in _features:
        if _data_health_check_results[feature]['invariant']:
            _invariant_features.append(feature)
            Log().log(msg=f'Feature ({feature}) is invariant')
        else:
            if _data_health_check_results[feature]['duplicated']:
                _duplicated_features.append(feature)
                Log().log(msg=f'Feature ({feature}) is duplicated')
            else:
                if feature in _relevant_features_to_check_missing_values:
                    if _data_health_check_results[feature]['missing_value_analysis']['proportion_of_missing_values'] < missing_value_threshold:
                        _valid_features.append(feature)
                        if _data_health_check_results[feature]['missing_value_analysis']['number_of_missing_values'] > 0:
                            _numeric_features_containing_missing_data.append(feature)
                    else:
                        Log().log(msg=f'Feature ({feature}) consists of {_data_health_check_results[feature]["missing_value_analysis"]["proportion_of_missing_values"] * 100}% of missing values which exceed missing value threshold of {missing_value_threshold * 100}%')
                else:
                    _valid_features.append(feature)
    _n_valid_features: int = len(_valid_features)
    if _n_valid_features == 0:
        _prop_valid_features: float = 0.0
    else:
        _prop_valid_features: float = round(number=_n_valid_features / len(_features), ndigits=4)
    for file_path, obj in [(output_file_path_missing_data, _numeric_features_containing_missing_data),
                           (output_file_path_invariant_features, _invariant_features),
                           (output_file_path_duplicated_features, _duplicated_features),
                           (output_file_path_valid_features, _valid_features),
                           (output_file_path_prop_valid_features, _prop_valid_features),
                           (output_file_path_n_valid_features, _n_valid_features),
                           ]:
        file_handler(file_path=file_path, obj=obj)
    _s3_output_file_path_data_health_check: str = s3_output_file_path_data_health_check
    if _s3_output_file_path_data_health_check is not None:
        _data_health_check_output: dict = dict(missing_data=_numeric_features_containing_missing_data,
                                               valid_features=_valid_features,
                                               prop_valid_features=_prop_valid_features,
                                               n_valid_features=_n_valid_features
                                               )
        if parallel_mode:
            _suffix: str = data_set_path.split('.')[0].split('_')[-1]
            _s3_output_file_path_data_health_check: str = s3_output_file_path_data_health_check.replace('.', f'_{_suffix}.')
        save_file_to_s3(file_path=_s3_output_file_path_data_health_check, obj=_data_health_check_output)
        Log().log(msg=f'Save data health check: {_s3_output_file_path_data_health_check}')
    _cpu_utilization: float = get_cpu_utilization(interval=1, logging=True)
    _cpu_utilization_per_cpu: List[float] = get_cpu_utilization_per_core(interval=1, logging=True)
    _memory_utilization: float = get_memory_utilization(logging=True)
    _memory_available = get_memory(total=False, logging=True)
    return [_numeric_features_containing_missing_data,
            _invariant_features,
            _duplicated_features,
            _valid_features,
            _prop_valid_features,
            _n_valid_features
            ]


if __name__ == '__main__':
    if ARGS.features:
        ARGS.features = ast.literal_eval(ARGS.features[0])
    data_health_check(data_set_path=ARGS.data_set_path,
                      analytical_data_types_path=ARGS.analytical_data_types_path,
                      output_file_path_missing_data=ARGS.output_file_path_missing_data,
                      output_file_path_invariant_features=ARGS.output_file_path_invariant_features,
                      output_file_path_duplicated_features=ARGS.output_file_path_duplicated_features,
                      output_file_path_valid_features=ARGS.output_file_path_valid_features,
                      output_file_path_prop_valid_features=ARGS.output_file_path_prop_valid_features,
                      output_file_path_n_valid_features=ARGS.output_file_path_n_valid_features,
                      features=ARGS.features,
                      missing_value_threshold=ARGS.missing_value_threshold,
                      parallel_mode=bool(ARGS.parallel_mode),
                      sep=ARGS.sep,
                      s3_output_file_path_data_health_check=ARGS.s3_output_file_path_data_health_check
                      )
