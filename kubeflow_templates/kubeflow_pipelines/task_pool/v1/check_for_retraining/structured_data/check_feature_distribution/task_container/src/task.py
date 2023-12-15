"""
Task: ... (Function to run in container)
"""

import argparse
import pandas as pd

from aws import file_exists, load_file_from_s3, save_file_to_s3
from datetime import datetime
from feature_distribution import FeatureDistribution
from file_handler import file_handler
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
    _df_current: pd.DataFrame = pd.read_csv(filepath_or_buffer=data_set_path, sep=sep)
    if file_exists(file_path=stats_file_path):
        _stats_obj: Dict[str, Dict[str, str]] = load_file_from_s3(file_path=stats_file_path)
        _n_previous_files: int = len(list(_stats_obj.keys()))
        _previous_file_path: str = _stats_obj[str(_n_previous_files - 1)]['file_path']
        _df_previous: pd.DataFrame = pd.read_csv(filepath_or_buffer=_previous_file_path, sep=sep)
        _analytical_data_type: Dict[str, List[str]] = load_file_from_s3(file_path=analytical_data_type_file_path)
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
    for file_path, obj in [(output_file_path_proceed, _proceed),
                           (output_file_path_msg, _msg),
                           (output_file_path_changed_features, _changed_features)
                           ]:
        file_handler(file_path=file_path, obj=obj)
    save_file_to_s3(file_path=stats_file_path, obj=_stats_obj)
    return [_proceed,
            _msg,
            _changed_features
            ]


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
