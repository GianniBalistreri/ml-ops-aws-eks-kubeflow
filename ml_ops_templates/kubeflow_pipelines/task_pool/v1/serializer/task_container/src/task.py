"""

Task: ... (Function to run in container)

"""

import argparse
import ast

from file_handler import file_handler
from resource_metrics import get_available_cpu, get_cpu_utilization, get_cpu_utilization_per_core, get_memory, get_memory_utilization
from serializer import Serializer, SerializerException
from typing import List, NamedTuple


PARSER = argparse.ArgumentParser(description="serialize data")
PARSER.add_argument('-action', type=str, required=True, default=None, help='distribution action')
PARSER.add_argument('-parallelized_obj', nargs='+', required=True, default=None, help='list of objects used in parallelization process')
PARSER.add_argument('-label_feature_name', type=str, required=False, default=None, help='name of the label feature containing given labels')
PARSER.add_argument('-labels', nargs='+', required=False, default=None, help='list of labels used in to identify origin of the cases to serialize')
PARSER.add_argument('-sep', type=str, required=False, default=',', help='column separator')
PARSER.add_argument('-output_file_path_missing_data', type=str, required=False, default=None, help='file path of features containing too much missing data output')
PARSER.add_argument('-output_file_path_valid_features', type=str, required=False, default=None, help='file path of valid features output')
PARSER.add_argument('-output_file_path_prop_valid_features', type=str, required=False, default=None, help='file path of the proportion of valid features output')
PARSER.add_argument('-output_file_path_n_valid_features', type=str, required=False, default=None, help='file path of number of valid features output')
PARSER.add_argument('-output_file_path_predictors', type=str, required=False, default=None, help='file path of the predictors output')
PARSER.add_argument('-output_file_path_new_target_feature', type=str, required=False, default=None, help='file path of the new target feature output')
PARSER.add_argument('-s3_output_file_path_parallelized_data', type=str, required=False, default=None, help='S3 file path of the serialized data output')
ARGS = PARSER.parse_args()


def serializer(action: str,
               parallelized_obj: list,
               label_feature_name: str = None,
               labels: list = None,
               output_file_path_missing_data: str = None,
               output_file_path_valid_features: str = None,
               output_file_path_prop_valid_features: str = None,
               output_file_path_n_valid_features: str = None,
               output_file_path_predictors: str = None,
               output_file_path_new_target_feature: str = None,
               s3_output_file_path_parallelized_data: str = None,
               sep: str = ','
               ) -> NamedTuple('outputs', [('missing_data', list),
                                           ('valid_features', list),
                                           ('prop_valid_features', float),
                                           ('n_valid_features', int),
                                           ('predictors', list),
                                           ('new_target_feature', str)
                                           ]
                               ):
    """
    Serialize data

    :param action: str
        Name of the distribution action
            -> cases: cases to new data set
            -> features: features to new data set
            -> analytical_data_types: distributed analytical data types
            -> processor_memory: distributed processor memory
            -> data_health_check: distributed results of data health check
            -> evolutionary_algorithm: distributed results of ml model optimization

    :param parallelized_obj: list
        List of objects used in parallelization process

    :param label_feature_name: str
            Name of the label feature that contains given labels

    :param labels: list
        Labels used in to identify origin of the cases to serialize

    :param output_file_path_missing_data: str
        Path of the features containing too much missing data output

    :param output_file_path_valid_features: str
        Path of the valid features output

    :param output_file_path_prop_valid_features: str
        Path of the proportion of valid features output

    :param output_file_path_n_valid_features: str
        Path of the number of valid features output

    :param output_file_path_predictors: str
        Path of the predictors output

    :param output_file_path_new_target_feature: str
        Path of the new target feature output

    :param s3_output_file_path_parallelized_data: str
        Complete file path of the parallelized data to save

    :param sep: str
        Separator

    :return: NamedTuple
        Serialized values
    """
    _cpu_available: int = get_available_cpu(logging=True)
    _memory_total: float = get_memory(total=True, logging=True)
    _memory_available: float = get_memory(total=False, logging=True)
    _missing_data: List[str] = []
    _valid_features: List[str] = []
    _prop_valid_features: float = None
    _n_valid_features: int = None
    _features: List[str] = []
    _new_target_feature: str = None
    _serializer: Serializer = Serializer(file_paths=parallelized_obj,
                                         output_file_path=s3_output_file_path_parallelized_data,
                                         contents=parallelized_obj,
                                         label_feature_name=label_feature_name,
                                         labels=labels,
                                         sep=sep
                                         )
    if action == 'analytical_data_types':
        _serializer.serialize_analytical_data_types()
    elif action == 'cases':
        _serializer.serialize_cases()
    elif action == 'features':
        _features = _serializer.serialize_features()
    elif action == 'data_health_check':
        _data_health_check: dict = _serializer.serialize_data_health_check_results()
        _missing_data = _data_health_check.get('missing_data')
        _valid_features = _data_health_check.get('valid_features')
        _prop_valid_features = _data_health_check.get('prop_valid_features')
        _n_valid_features = _data_health_check.get('n_valid_features')
    elif action == 'evolutionary_algorithm':
        _serializer.serialize_evolutionary_results()
    elif action == 'processor_memory':
        _processing_memory: dict = _serializer.serialize_processor_memory()
        _features = _processing_memory.get('predictors')
        _new_target_feature = _processing_memory.get('new_target_feature')
    else:
        raise SerializerException(f'Action ({action}) not supported')
    if output_file_path_missing_data is not None:
        file_handler(file_path=output_file_path_missing_data, obj=_missing_data)
    if output_file_path_valid_features is not None:
        file_handler(file_path=output_file_path_valid_features, obj=_valid_features)
    if output_file_path_prop_valid_features is not None:
        file_handler(file_path=output_file_path_prop_valid_features, obj=_prop_valid_features)
    if output_file_path_n_valid_features is not None:
        file_handler(file_path=output_file_path_n_valid_features, obj=_n_valid_features)
    if output_file_path_predictors is not None:
        file_handler(file_path=output_file_path_predictors, obj=_features)
    if output_file_path_new_target_feature is not None:
        file_handler(file_path=output_file_path_new_target_feature, obj=_new_target_feature)
    _cpu_utilization: float = get_cpu_utilization(interval=1, logging=True)
    _cpu_utilization_per_cpu: List[float] = get_cpu_utilization_per_core(interval=1, logging=True)
    _memory_utilization: float = get_memory_utilization(logging=True)
    _memory_available = get_memory(total=False, logging=True)
    return [_missing_data,
            _valid_features,
            _prop_valid_features,
            _n_valid_features,
            _features,
            _new_target_feature
            ]


if __name__ == '__main__':
    if ARGS.parallelized_obj:
        ARGS.parallelized_obj = ast.literal_eval(ARGS.parallelized_obj[0])
    if ARGS.labels:
        ARGS.labels = ast.literal_eval(ARGS.labels[0])
    serializer(action=ARGS.action,
               parallelized_obj=ARGS.parallelized_obj,
               label_feature_name=ARGS.label_feature_name,
               labels=ARGS.labels,
               output_file_path_missing_data=ARGS.output_file_path_missing_data,
               output_file_path_valid_features=ARGS.output_file_path_valid_features,
               output_file_path_prop_valid_features=ARGS.output_file_path_prop_valid_features,
               output_file_path_n_valid_features=ARGS.output_file_path_n_valid_features,
               output_file_path_predictors=ARGS.output_file_path_predictors,
               output_file_path_new_target_feature=ARGS.output_file_path_new_target_feature,
               s3_output_file_path_parallelized_data=ARGS.s3_output_file_path_parallelized_data,
               sep=ARGS.sep
               )
