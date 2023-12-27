"""

Task: ... (Function to run in container)

"""

import argparse

from aws import save_file_to_s3
from custom_logger import Log
from file_handler import file_handler
from parallelizer import Parallelizer
from typing import Any, Dict, List, NamedTuple


PARSER = argparse.ArgumentParser(description="parallelize data")
PARSER.add_argument('-action', type=str, required=True, default=None, help='distribution action')
PARSER.add_argument('-analytical_data_types', type=Any, required=False, default=None, help='pre-defined analytical data types')
PARSER.add_argument('-data_file_path', type=str, required=False, default=None, help='complete file path of the input data set')
PARSER.add_argument('-s3_bucket_name', type=str, required=False, default=None, help='name of the S3 bucket')
PARSER.add_argument('-chunks', type=int, required=False, default=4, help='number of chunks to distribute')
PARSER.add_argument('-persist_data', type=int, required=False, default=1, help='whether to persist distributed chunks or not')
PARSER.add_argument('-elements', type=list, required=False, default=None, help='elements to distribute')
PARSER.add_argument('-prefix', type=str, required=False, default=None, help='prefix used for filtering folders in S3 bucket')
PARSER.add_argument('-sep', type=str, required=False, default=',', help='column separator')
PARSER.add_argument('-output_path_distribution', type=str, required=False, default=None, help='file path of the distribution output')
PARSER.add_argument('-s3_output_path_distribution', type=str, required=False, default=None, help='S3 file path of the distribution output')
ARGS = PARSER.parse_args()


def parallelizer(action: str,
                 output_path_distribution: str,
                 analytical_data_types: Dict[str, List[str]] = None,
                 data_file_path: str = None,
                 s3_bucket_name: str = None,
                 chunks: int = 4,
                 persist_data: bool = True,
                 elements: list = None,
                 prefix: str = None,
                 sep: str = ',',
                 s3_output_path_distribution: str = None
                 ) -> NamedTuple('outputs', [('distributed_values', list)]):
    """
    Parallelize data

    :param action: str
        Name of the distribution action
            -> cases: cases of given data set
            -> elements: elements of given list
            -> features: features of given data set
            -> file_paths: file path in given S3 bucket

    :param output_path_distribution: str
        Path of the distribution output

    :param analytical_data_types: dict
        Assigned analytical data types to each feature

    :param data_file_path: str
        Complete file path of the data set

    :param s3_bucket_name: str
        Name of the S3 bucket

    :param chunks: int
        Number of chunks to distribute

    :param persist_data: bool
        Whether to persist distributed chunks or not

    :param elements: list
        Elements to distribute

    :param prefix: str
        Prefix used for filtering folder in S3 bucket

    :param sep: str
        Separator

    :param s3_output_path_distribution: str
        Complete file path of the distribution output

    :return: NamedTuple
        Distributed values
    """
    _parallelizer: Parallelizer = Parallelizer(file_path=data_file_path,
                                               chunks=chunks,
                                               persist_data=persist_data,
                                               analytical_data_types=analytical_data_types,
                                               elements=elements,
                                               s3_bucket_name=s3_bucket_name,
                                               prefix=prefix,
                                               sep=sep
                                               )
    _distributed_values: list = _parallelizer.main(action=action)
    file_handler(file_path=output_path_distribution, obj=_distributed_values)
    if s3_output_path_distribution is not None:
        save_file_to_s3(file_path=s3_output_path_distribution, obj=_distributed_values)
        Log().log(msg=f'Save distribution: {s3_output_path_distribution}')
    return [_distributed_values]


if __name__ == '__main__':
    parallelizer(action=ARGS.action,
                 output_path_distribution=ARGS.output_path_distribution,
                 analytical_data_types=ARGS.analytical_data_types,
                 data_file_path=ARGS.data_file_path,
                 s3_bucket_name=ARGS.s3_bucket_name,
                 chunks=ARGS.chunks,
                 persist_data=ARGS.persist_data,
                 elements=ARGS.elements,
                 prefix=ARGS.prefix,
                 sep=ARGS.sep,
                 s3_output_path_distribution=ARGS.s3_output_path_distribution
                 )
