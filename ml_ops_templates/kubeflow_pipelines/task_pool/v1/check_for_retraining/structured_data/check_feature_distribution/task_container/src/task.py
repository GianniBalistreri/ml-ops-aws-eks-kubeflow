"""
Task: ... (Function to run in container)
"""

import argparse
import pandas as pd

from aws import file_exists, load_file_from_s3, load_file_from_s3_as_df, save_file_to_s3
from custom_logger import Log
from datetime import datetime
from feature_distribution import FeatureDistribution
from file_handler import file_handler
from resource_metrics import get_available_cpu, get_cpu_utilization, get_cpu_utilization_per_core, get_memory, get_memory_utilization
from typing import Dict, NamedTuple, List


PARSER = argparse.ArgumentParser(description="check feature distribution")
PARSER.add_argument('-data_set_path', type=str, required=True, default=None, help='file path of the data set')
PARSER.add_argument('-analytical_data_type_file_path', type=str, required=True, default=None, help='file path of the analytical data type')
PARSER.add_argument('-output_file_path_proceed', type=str, required=True, default=None, help='complete customized file path of the proceed output')
PARSER.add_argument('-output_file_path_msg', type=str, required=True, default=None, help='complete customized file path of the messge output')
PARSER.add_argument('-output_file_path_changed_features', type=str, required=True, default=None, help='complete customized file path of the changed features output')
PARSER.add_argument('-sep', type=str, required=False, default=',', help='column separator')
ARGS = PARSER.parse_args()


def check_feature_distribution(data_set_path: str,
                               stats_file_path: str,
                               analytical_data_type_file_path: str,
                               output_file_path_proceed: str,
                               output_file_path_msg: str,
                               output_file_path_changed_features: str,
                               categorical_test_meth: str = 'chi2_test',
                               continuous_test_meth: str = 'mann_whitney_u_test',
                               sep: str = ','
                               ) -> NamedTuple('outputs', [('proceed', bool),
                                                           ('msg', str),
                                                           ('changed_features', list)
                                                           ]
                                               ):
    """
    Check feature distributions of structured (tabular) data for significant changes to use it for continuous training approach

    :param data_set_path: str
        Complete file path of the data set

    :param stats_file_path: str
        Complete file path of the calculated feature distribution statistics

    :param analytical_data_type_file_path: str
        Complete file path of the analytical data types

    :param output_file_path_proceed: str
        Complete file path of the proceed output file

    :param output_file_path_msg: str
        Complete file path of the message output file

    :param output_file_path_changed_features: str
        Complete file path of the changed features output file

    :param categorical_test_meth: str
        Name of the statistical test method for categorical features

    :param continuous_test_meth: str
        Name of the statistical test method for continuous features

    :param sep: str
        Separator

    :return: NamedTuple
        Whether to proceed with pipeline processes and message if no significant changes are detected
    """
    _cpu_available: int = get_available_cpu(logging=True)
    _memory_total: float = get_memory(total=True, logging=True)
    _memory_available: float = get_memory(total=False, logging=True)
    _df_current: pd.DataFrame = load_file_from_s3_as_df(file_path=data_set_path, sep=sep)
    Log().log(msg=f'Load data set: {data_set_path} -> Cases={_df_current.shape[0]}, Features={_df_current.shape[1]}')
    if file_exists(file_path=stats_file_path):
        _stats_obj: Dict[str, Dict[str, str]] = load_file_from_s3(file_path=stats_file_path)
        _n_previous_files: int = len(list(_stats_obj.keys()))
        _previous_file_path: str = _stats_obj[str(_n_previous_files - 1)]['file_path']
        _df_previous: pd.DataFrame = load_file_from_s3_as_df(file_path=_previous_file_path, sep=sep)
        Log().log(msg=f'Load previous data set: {data_set_path} -> Cases={_df_previous.shape[0]}, Features={_df_previous.shape[1]}')
        _analytical_data_type: Dict[str, List[str]] = load_file_from_s3(file_path=analytical_data_type_file_path)
        Log().log(msg=f'Load analytical data types: {analytical_data_type_file_path}')
        _changed_features: List[str] = []
        for analytical_data_type in _analytical_data_type.keys():
            if analytical_data_type == 'categorical':
                for cat in _analytical_data_type.get(analytical_data_type):
                    _feature_distribution: FeatureDistribution = FeatureDistribution(previous_observation_values=_df_previous[cat],
                                                                                     previous_observation_name=cat,
                                                                                     current_observation_values=_df_current[cat],
                                                                                     current_observation_name=cat,
                                                                                     bootstrapping_n_samples=10,
                                                                                     bootstrapping_replace=False,
                                                                                     p=0.95
                                                                                     )
                    _reject: bool = _feature_distribution.main(meth=categorical_test_meth)
                    if _reject:
                        _changed_features.append(cat)
                        Log().log(msg=f'Univariate distribution of categorical feature ({cat}) changed significantly')
            elif analytical_data_type == 'continuous':
                for num in _analytical_data_type.get(analytical_data_type):
                    _feature_distribution: FeatureDistribution = FeatureDistribution(previous_observation_values=_df_previous[num],
                                                                                     previous_observation_name=num,
                                                                                     current_observation_values=_df_current[num],
                                                                                     current_observation_name=num,
                                                                                     bootstrapping_n_samples=10,
                                                                                     bootstrapping_replace=False,
                                                                                     p=0.95
                                                                                     )
                    _reject: bool = _feature_distribution.main(meth=continuous_test_meth)
                    if _reject:
                        _changed_features.append(num)
                        Log().log(msg=f'Univariate distribution of continuous feature ({num}) changed significantly')
    else:
        _stats_obj: Dict[str, Dict[str, str]] = {}
        _n_previous_files: int = 0
        _changed_features: List[str] = _df_current.columns.tolist()
    _stats_obj.update({str(_n_previous_files): dict(time=str(datetime.now()), file_path=data_set_path)})
    _proceed: bool = len(_changed_features) > 0
    if _proceed:
        _msg: str = 'Significant changes in feature distribution detected'
    else:
        _msg: str = 'No significant changes in feature distribution detected'
    save_file_to_s3(file_path=stats_file_path, obj=_stats_obj)
    Log().log(msg=f'Save distribution statistics: {stats_file_path}')
    file_handler(file_path=output_file_path_proceed, obj=_proceed)
    file_handler(file_path=output_file_path_msg, obj=_msg)
    file_handler(file_path=output_file_path_changed_features, obj=_changed_features)
    _cpu_utilization: float = get_cpu_utilization(interval=1, logging=True)
    _cpu_utilization_per_cpu: List[float] = get_cpu_utilization_per_core(interval=1, logging=True)
    _memory_utilization: float = get_memory_utilization(logging=True)
    _memory_available = get_memory(total=False, logging=True)
    return [_proceed, _msg, _changed_features]


if __name__ == '__main__':
    check_feature_distribution(data_set_path=ARGS.data_set_path,
                               stats_file_path=ARGS.stats_file_path,
                               analytical_data_type_file_path=ARGS.analytical_data_type_file_path,
                               output_file_path_proceed=ARGS.output_file_path_proceed,
                               output_file_path_msg=ARGS.output_file_path_msg,
                               output_file_path_changed_features=ARGS.output_file_path_changed_features,
                               categorical_test_meth=ARGS.categorical_test_meth,
                               continuous_test_meth=ARGS.continuous_test_meth,
                               sep=ARGS.sep
                               )
