"""

Task: ... (Function to run in container)

"""

import argparse

from file_handler import file_handler
from serializer import Serializer
from typing import NamedTuple


PARSER = argparse.ArgumentParser(description="serialize data")
PARSER.add_argument('-action', type=str, required=True, default=None, help='distribution action')
PARSER.add_argument('-parallelized_obj', type=str, required=True, default=None, help='list of objects used in parallelization process')
PARSER.add_argument('-sep', type=str, required=False, default=',', help='column separator')
PARSER.add_argument('-output_file_path', type=str, required=True, default=None, help='complete file path of the data output')
PARSER.add_argument('-output_file_path_serialization', type=str, required=True, default=None, help='file path of the serialization output')
ARGS = PARSER.parse_args()


def serializer(action: str,
               parallelized_obj: list,
               output_file_path: str,
               output_file_path_serialization: str,
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

    :param output_file_path_serialization: str
        Path of the serialization output

    :param sep: str
        Separator

    :return: NamedTuple
        Serialized values
    """
    _serializer: Serializer = Serializer(file_paths=parallelized_obj,
                                         output_file_path=output_file_path,
                                         contents=parallelized_obj,
                                         sep=sep
                                         )
    _serialized_values: dict = _serializer.main(action=action)
    file_handler(file_path=output_file_path_serialization, obj=_serialized_values)
    return [_serialized_values]


if __name__ == '__main__':
    serializer(action=ARGS.action,
               parallelized_obj=ARGS.parallelized_obj,
               output_file_path=ARGS.output_file_path,
               output_file_path_serialization=ARGS.output_file_path_serialization,
               sep=ARGS.sep
               )
