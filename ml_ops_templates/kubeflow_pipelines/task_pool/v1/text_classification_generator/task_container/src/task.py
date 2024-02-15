"""

Task: ... (Function to run in container)

"""

import argparse
import pandas as pd

from aws import save_file_to_s3
from custom_logger import Log
from text_classfication_generator import DataHealthCheck
from file_handler import file_handler
from typing import Any, List, NamedTuple


PARSER = argparse.ArgumentParser(description="check data health")
PARSER.add_argument('-data_set_path', type=str, required=True, default=None, help='file path of the data set')
PARSER.add_argument('-analytical_data_types', type=Any, required=True, default=None, help='assignment of features to analytical data types')
PARSER.add_argument('-missing_value_threshold', type=float, required=False, default=0.95, help='threshold to classify features as invalid based on the amount of missing values')
PARSER.add_argument('-sep', type=str, required=False, default=',', help='column separator')
PARSER.add_argument('-output_file_path_data_missing_data', type=str, required=True, default=None, help='file path of features containing too much missing data output')
PARSER.add_argument('-output_file_path_invariant', type=str, required=True, default=None, help='file path of invariant features output')
PARSER.add_argument('-output_file_path_duplicated', type=str, required=True, default=None, help='file path of duplicated features output')
PARSER.add_argument('-output_file_path_valid_features', type=str, required=True, default=None, help='file path of valid features output')
PARSER.add_argument('-output_file_path_prop_valid_features', type=str, required=True, default=None, help='file path of the proportion of valid features output')
PARSER.add_argument('-s3_output_file_path_data_health_check', type=str, required=False, default=None, help='S3 file path of the data health check output')
ARGS = PARSER.parse_args()


def model_generator(data_set_path: str,
                      analytical_data_types: dict,
                      output_file_path_missing_data: str,
                      output_file_path_invariant: str,
                      output_file_path_duplicated: str,
                      output_file_path_valid_features: str,
                      output_file_path_prop_valid_features: str,
                      missing_value_threshold: float = 0.95,
                      sep: str = ',',
                      s3_output_file_path_data_health_check: str = None
                      ) -> NamedTuple('outputs', [('missing_data', List[str]),
                                                  ('invariant', List[str]),
                                                  ('duplicated', List[str]),
                                                  ('valid_features', List[str]),
                                                  ('prop_valid_features', float)
                                                  ]
                                      ):
    """
    Examine data health of structured (tabular) data

    :param data_set_path: str
        Complete file path of the data set

    :param analytical_data_types: dict
        Assigned analytical data types to each feature

    :param output_file_path_missing_data: str
        Path of the features containing too much missing data output

    :param output_file_path_invariant: str
        Path of the invariant features output

    :param output_file_path_duplicated: str
        Path of the duplicated features output

    :param output_file_path_valid_features: str
        Path of the valid features output

    :param output_file_path_prop_valid_features: str
        Path of the proportion of valid features output

    :param missing_value_threshold: float
        Threshold of missing values to exclude numeric feature

    :param sep: str
        Separator

    :param s3_output_file_path_data_health_check: str
        Complete file path of the data health check output

    :return: NamedTuple
        Data health check of given features, invalid and valid features, proportion of valid features
    """
    _df: pd.DataFrame = pd.read_csv(filepath_or_buffer=data_set_path, sep=sep)

    for file_path, obj in [(output_file_path_missing_data, _numeric_features_containing_missing_data),
                           (output_file_path_invariant, _invariant_features),
                           (output_file_path_duplicated, _duplicated_features),
                           (output_file_path_valid_features, _valid_features),
                           (output_file_path_prop_valid_features, _prop_valid_features)
                           ]:
        file_handler(file_path=file_path, obj=obj)
    if s3_output_file_path_data_health_check is not None:
        save_file_to_s3(file_path=s3_output_file_path_data_health_check, obj=_data_health_check_results)
        Log().log(msg=f'Save data health check: {s3_output_file_path_data_health_check}')
    return [_numeric_features_containing_missing_data,
            _invariant_features,
            _duplicated_features,
            _valid_features,
            _prop_valid_features
            ]


if __name__ == '__main__':
    model_generator(data_set_path=ARGS.data_set_path,
                      analytical_data_types=ARGS.analytical_data_types,
                      output_file_path_missing_data=ARGS.output_file_path_missing_data,
                      output_file_path_invariant=ARGS.output_file_path_invariant,
                      output_file_path_duplicated=ARGS.output_file_path_duplicated,
                      output_file_path_valid_features=ARGS.output_file_path_valid_features,
                      output_file_path_prop_valid_features=ARGS.output_file_path_prop_valid_features,
                      missing_value_threshold=ARGS.missing_value_threshold,
                      sep=ARGS.sep,
                      s3_output_file_path_data_health_check=ARGS.s3_output_file_path_data_health_check
                      )
