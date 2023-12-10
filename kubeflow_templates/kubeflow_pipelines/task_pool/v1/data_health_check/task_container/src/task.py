"""

Task: ... (Function to run in container)

"""

import argparse
import pandas as pd

from aws import save_file_to_s3
from data_health_check import DataHealthCheck
from file_handler import file_handler
from typing import Any, NamedTuple, List


PARSER = argparse.ArgumentParser(description="check data health")
PARSER.add_argument('-data_set_path', type=str, required=True, default=None, help='file path of the data set')
PARSER.add_argument('-analytical_data_types', type=Any, required=True, default=None, help='assignment of features to analytical data types')
PARSER.add_argument('-output_bucket_name', type=str, required=True, default=None, help='name of the S3 output bucket')
PARSER.add_argument('-output_file_path_data_health_check', type=str, required=True, default=None, help='file path of the data health check output')
PARSER.add_argument('-output_file_path_data_missing_data', type=str, required=True, default=None, help='file path of features containing too much missing data')
PARSER.add_argument('-output_file_path_invariant', type=str, required=True, default=None, help='file path of invariant features')
PARSER.add_argument('-output_file_path_duplicated', type=str, required=True, default=None, help='file path of duplicated features')
PARSER.add_argument('-output_file_path_valid_features', type=str, required=True, default=None, help='file path of valid features')
PARSER.add_argument('-output_file_path_prop_valid_features', type=str, required=True, default=None, help='file path of the proportion of valid features')
PARSER.add_argument('-sep', type=str, required=False, default=',', help='column separator')
PARSER.add_argument('-missing_value_threshold', type=float, required=False, default=0.8, help='threshold to classify features as invalid based on the amount of missing values')
PARSER.add_argument('-output_file_path_data_health_check_customized', type=str, required=False, default=None, help='complete customized file path of the data health check output')
PARSER.add_argument('-output_file_path_missing_data_customized', type=str, required=False, default=None, help='complete customized file path of the missing data output')
PARSER.add_argument('-output_file_path_invariant_customized', type=str, required=False, default=None, help='file path of invariant features')
PARSER.add_argument('-output_file_path_duplicated_customized', type=str, required=False, default=None, help='file path of duplicated features')
PARSER.add_argument('-output_file_path_valid_features_customized', type=str, required=False, default=None, help='file path of valid features')
PARSER.add_argument('-output_file_path_prop_valid_features_customized', type=str, required=False, default=None, help='file path of the proportion of valid features')
ARGS = PARSER.parse_args()


def data_health_check(data_set_path: str,
                      analytical_data_types: dict,
                      output_file_path_data_health_check: str,
                      output_file_path_missing_data: str,
                      output_file_path_invariant: str,
                      output_file_path_duplicated: str,
                      output_file_path_valid_features: str,
                      output_file_path_prop_valid_features: str,
                      sep: str = ',',
                      missing_value_threshold: float = 0.8,
                      output_file_path_data_health_check_customized: str = None,
                      output_file_path_missing_data_customized: str = None,
                      output_file_path_invariant_customized: str = None,
                      output_file_path_duplicated_customized: str = None,
                      output_file_path_valid_features_customized: str = None,
                      output_file_path_prop_valid_features_customized: str = None
                      ) -> NamedTuple('outputs', [('data_health_check', dict),
                                                  ('missing_data', List[str]),
                                                  ('invariant', List[str]),
                                                  ('duplicated', List[str]),
                                                  ('valid_features', List[str]),
                                                  ('prop_valid_features', float)
                                                  ]
                                      ):
    """
    Evaluate data health of structured (tabular) data

    :param data_set_path: str
        Complete file path of the data set

    :param analytical_data_types: dict
        Assigned analytical data types to each feature

    :param output_file_path_data_health_check: str
        Path of the data health check results

    :param output_file_path_missing_data: str
        Path of the features containing too much missing data

    :param output_file_path_invariant: str
        Path of the invariant features

    :param output_file_path_duplicated: str
        Path of the duplicated features

    :param output_file_path_valid_features: str
        Path of the valid features

    :param output_file_path_prop_valid_features: str
        Path of the proportion of valid features

    :param sep: str
        Separator

    :param missing_value_threshold: float
        Threshold of missing values to exclude numeric feature

    :param output_file_path_data_health_check_customized: str
        Complete customized file path of the data health check output

    :param output_file_path_missing_data_customized: str
        Complete customized file path of the missing data analysis output

    :param output_file_path_invariant_customized: str
        Complete customized file path of the invariant features output

    :param output_file_path_duplicated_customized: str
        Complete customized file path of the duplicated features output

    :param output_file_path_valid_features_customized: str
        Complete customized file path of the valid features output

    :param output_file_path_prop_valid_features_customized: str
        Complete customized file path of the proportion of valid features output

    :return: NamedTuple
        Data health check of given features, invalid and valid features, proportion of valid features
    """
    _df: pd.DataFrame = pd.read_csv(filepath_or_buffer=data_set_path, sep=sep)
    _feature_names: List[str] = _df.columns.tolist()
    _data_health_check: DataHealthCheck = DataHealthCheck(df=_df, feature_names=_feature_names)
    _data_health_check_results: dict = _data_health_check.main()
    _relevant_features_to_check_missing_values: List[str] = analytical_data_types.get('continuous')
    _relevant_features_to_check_missing_values.extend(analytical_data_types.get('date'))
    _valid_features: List[str] = []
    _invariant_features: List[str] = []
    _duplicated_features: List[str] = []
    _numeric_features_containing_missing_data: List[str] = []
    for feature in _feature_names:
        if _data_health_check_results[feature]['invariant']:
            _invariant_features.append(feature)
        else:
            if _data_health_check_results[feature]['duplicated']:
                _duplicated_features.append(feature)
            else:
                if feature in _relevant_features_to_check_missing_values:
                    if _data_health_check_results[feature]['missing_value_analysis']['proportion_of_missing_values'] < missing_value_threshold:
                        _valid_features.append(feature)
                    if _data_health_check_results[feature]['missing_value_analysis']['number_of_missing_values'] > 0:
                        _numeric_features_containing_missing_data.append(feature)
                else:
                    _valid_features.append(feature)
    if len(_valid_features) == 0:
        _prop_valid_features: float = 0.0
    else:
        _prop_valid_features: float = round(number=len(_valid_features) / len(_feature_names), ndigits=4)
    for file_path, obj, customized_file_path in [(output_file_path_data_health_check, _data_health_check_results, output_file_path_data_health_check_customized),
                                                 (output_file_path_missing_data, _numeric_features_containing_missing_data, output_file_path_missing_data_customized),
                                                 (output_file_path_invariant, _invariant_features, output_file_path_invariant_customized),
                                                 (output_file_path_duplicated, _duplicated_features, output_file_path_duplicated_customized),
                                                 (output_file_path_valid_features, _valid_features, output_file_path_valid_features_customized),
                                                 (output_file_path_prop_valid_features, _prop_valid_features, output_file_path_prop_valid_features_customized)
                                                 ]:
        file_handler(file_path=file_path, obj=obj)
        if customized_file_path is not None:
            save_file_to_s3(file_path=customized_file_path, obj=obj)
    return [_data_health_check_results,
            _numeric_features_containing_missing_data,
            _invariant_features,
            _duplicated_features,
            _valid_features,
            _prop_valid_features
            ]


if __name__ == '__main__':
    data_health_check(data_set_path=ARGS.data_set_path,
                      analytical_data_types=ARGS.analytical_data_types,
                      output_file_path_data_health_check=ARGS.output_file_path_data_health_check,
                      output_file_path_missing_data=ARGS.output_file_path_missing_data,
                      output_file_path_invariant=ARGS.output_file_path_invariant,
                      output_file_path_duplicated=ARGS.output_file_path_duplicated,
                      output_file_path_valid_features=ARGS.output_file_path_valid_features,
                      output_file_path_prop_valid_features=ARGS.output_file_path_prop_valid_features,
                      sep=ARGS.sep,
                      missing_value_threshold=ARGS.missing_value_threshold,
                      output_file_path_data_health_check_customized=ARGS.output_file_path_data_health_check_customized,
                      output_file_path_missing_data_customized=ARGS.output_file_path_missing_data_customized,
                      output_file_path_invariant_customized=ARGS.output_file_path_invariant_customized,
                      output_file_path_duplicated_customized=ARGS.output_file_path_duplicated_customized,
                      output_file_path_valid_features_customized=ARGS.output_file_path_valid_features_customized,
                      output_file_path_prop_valid_features_customized=ARGS.output_file_path_prop_valid_features_customized
                      )
