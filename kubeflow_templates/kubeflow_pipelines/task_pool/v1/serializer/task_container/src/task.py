"""

Task: ... (Function to run in container)

"""

import argparse

from file_handler import file_handler
from serializer import serialize_cases, serialize_features, serialize_evolutionary_results, SerializerException
from typing import NamedTuple


PARSER = argparse.ArgumentParser(description="serialize data")
PARSER.add_argument('-action', type=str, required=True, default=None, help='distribution action')
PARSER.add_argument('-parallelized_obj', type=str, required=True, default=None, help='list of objects used in parallelization process')
PARSER.add_argument('-output_file_path', type=str, required=True, default=None, help='complete file path of the data output')
PARSER.add_argument('-sep', type=str, required=False, default=',', help='column separator')
ARGS = PARSER.parse_args()


def serializer(action: str,
               parallelized_obj: list,
               output_file_path: str,
               sep: str = ','
               ) -> NamedTuple('outputs', [('serialized_values', list)]):
    """
    Serialize data

    :param action: str
        Name of the distribution action
            -> cases: cases to new data set
            -> features: features to new data set
            -> evolutionary_algorithm: distributed results of ml model optimization

    :param parallelized_obj: list
        List of objects used in parallelization process

    :param output_file_path: str
        Complete file path of the aggregated data

    :param sep: str
        Separator

    :return: NamedTuple
        Serialized values
    """
    _serialized_values: dict = None
    if action == 'cases':
        serialize_cases(file_paths=parallelized_obj, output_file_path=output_file_path, sep=sep)
    elif action == 'features':
        serialize_features(file_paths=parallelized_obj, output_file_path=output_file_path, sep=sep)
    elif action == 'evolutionary_algorithm':
        _serialized_values = serialize_evolutionary_results(contents=parallelized_obj)
    else:
        raise SerializerException(f'Action ({action}) not supported')
    file_handler(file_path='serialized_values.json', obj=_serialized_values)
    return [_serialized_values]


if __name__ == '__main__':
    serializer(action=ARGS.action,
               parallelized_obj=ARGS.parallelized_obj,
               output_file_path=ARGS.output_file_path,
               sep=ARGS.sep
               )
