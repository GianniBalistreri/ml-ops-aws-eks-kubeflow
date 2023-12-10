"""

Task: ... (Function to run in container)

"""

import argparse
import json

from parallelizer import distribute_cases, distribute_elements, distribute_features, distribute_file_paths, ParallelizerException
from typing import NamedTuple


PARSER = argparse.ArgumentParser(description="parallelize data")
PARSER.add_argument('-action', type=str, required=True, default=None, help='distribution action')
PARSER.add_argument('-data_file_path', type=str, required=False, default=None, help='complete file path of the input data set')
PARSER.add_argument('-bucket_name', type=str, required=False, default=None, help='name of the S3 bucket')
PARSER.add_argument('-sep', type=str, required=False, default=',', help='column separator')
ARGS = PARSER.parse_args()


def parallelizer(action: str,
                 data_file_path: str = None,
                 bucket_name: str = None,
                 sep: str = ','
                 ) -> NamedTuple('outputs', [('distributed_values', list)]):
    """
    Parallelize data

    :param action: str
        Name of the distribution action
            -> cases: cases of given data set
            -> elements: elements of given list
            -> features: features of given data set
            -> file_paths: file path in given S3 bucket

    :param data_file_path: str
        Complete file path of the data set

    :param bucket_name: str
        Name of the S3 bucket

    :param sep: str
        Separator

    :return: NamedTuple
        Distributed values
    """
    if action == 'cases':
        _distributed_values: list = distribute_cases(file_path=data_file_path,
                                                     chunks=4,
                                                     persist_data=True,
                                                     sep=sep
                                                     )
    elif action == 'elements':
        _distributed_values: list = distribute_elements(elements=[],
                                                        chunks=5
                                                        )
    elif action == 'features':
        _distributed_values: list = distribute_features(file_path=data_file_path,
                                                        persist_data=True,
                                                        chunks=None,
                                                        sep=sep
                                                        )
    elif action == 'file_paths':
        _distributed_values: list = distribute_file_paths(chunks=4,
                                                          bucket_name=bucket_name,
                                                          prefix=None
                                                          )
    else:
        raise ParallelizerException(f'Action ({action}) not supported')
    with open('distributed_values.json', 'w') as _file:
        json.dump(_distributed_values, _file)
    return [_distributed_values]


if __name__ == '__main__':
    parallelizer(action=ARGS.action,
                 data_file_path=ARGS.data_file_path,
                 bucket_name=ARGS.bucket_name,
                 sep=ARGS.sep
                 )
