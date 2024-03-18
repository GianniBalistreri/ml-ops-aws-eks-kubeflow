"""

Task: ... (Function to run in container)

"""

import argparse
import ast

from aws import load_file_from_s3, save_file_to_s3
from custom_logger import Log
from file_handler import file_handler
from parallelizer import Parallelizer, ParallelizerException
from typing import Dict, List, NamedTuple


PARSER = argparse.ArgumentParser(description="parallelize data")
PARSER.add_argument('-action', type=str, required=True, default=None, help='distribution action')
PARSER.add_argument('-analytical_data_types_path', type=str, required=False, default=None, help='assignment of features to analytical data types')
PARSER.add_argument('-data_file_path', type=str, required=False, default=None, help='complete file path of the input data set')
PARSER.add_argument('-s3_bucket_name', type=str, required=False, default=None, help='name of the S3 bucket')
PARSER.add_argument('-chunks', type=int, required=False, default=4, help='number of chunks to distribute')
PARSER.add_argument('-persist_data', type=int, required=False, default=1, help='whether to persist distributed chunks or not')
PARSER.add_argument('-elements', nargs='+', required=False, default=None, help='elements to distribute')
PARSER.add_argument('-split_by', type=str, required=False, default=None, help='name of the label feature to split by')
PARSER.add_argument('-prefix', type=str, required=False, default=None, help='prefix used for filtering folders in S3 bucket')
PARSER.add_argument('-sep', type=str, required=False, default=',', help='column separator')
PARSER.add_argument('-output_path_distribution', type=str, required=False, default=None, help='file path of the distribution output')
PARSER.add_argument('-s3_output_path_distribution', type=str, required=False, default=None, help='S3 file path of the distribution output')
ARGS = PARSER.parse_args()


def parallelizer(action: str,
                 output_path_distribution: str,
                 analytical_data_types_path: str = None,
                 data_file_path: str = None,
                 s3_bucket_name: str = None,
                 chunks: int = 4,
                 persist_data: bool = True,
                 elements: list = None,
                 split_by: str = None,
                 prefix: str = None,
                 sep: str = ',',
                 s3_output_path_distribution: str = None
                 ) -> NamedTuple('outputs', [('distributed_values', list)]):
    """
    Parallelize data

    :param action: str
        Name of the distribution action
            -> analytical_data_types: features of given data set based on analytical data types assignment
            -> cases: cases of given data set
            -> elements: elements of given list
            -> features: features of given data set
            -> file_paths: file path in given S3 bucket

    :param output_path_distribution: str
        Path of the distribution output

    :param analytical_data_types_path: str
        Complete file path of the analytical data types

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

    :param split_by: str
            Name of the features to split cases by

    :param prefix: str
        Prefix used for filtering folder in S3 bucket

    :param sep: str
        Separator

    :param s3_output_path_distribution: str
        Complete file path of the distribution output

    :return: NamedTuple
        Distributed values
    """
    if analytical_data_types_path is None:
        _analytical_data_types: Dict[str, List[str]] = None
    else:
        _analytical_data_types: Dict[str, List[str]] = load_file_from_s3(file_path=analytical_data_types_path)
        Log().log(msg=f'Load analytical data types: {analytical_data_types_path}')
    _parallelizer: Parallelizer = Parallelizer(file_path=data_file_path,
                                               chunks=chunks,
                                               persist_data=persist_data,
                                               analytical_data_types=_analytical_data_types,
                                               elements=elements,
                                               split_by=split_by,
                                               s3_bucket_name=s3_bucket_name,
                                               prefix=prefix,
                                               sep=sep
                                               )
    if action == 'analytical_data_types':
        _distributed_values: list = _parallelizer.distribute_analytical_data_types()
    elif action == 'cases':
        _distributed_values: list = _parallelizer.distribute_cases()
    elif action == 'elements':
        _distributed_values: List[list] = _parallelizer.distribute_elements()
    elif action == 'features':
        _distributed_values: list = _parallelizer.distribute_features()
    elif action == 'file_paths':
        _distributed_values: List[List[str]] = _parallelizer.distribute_file_paths()
    else:
        raise ParallelizerException(f'Action ({action}) not supported')
    file_handler(file_path=output_path_distribution, obj=_distributed_values)
    if s3_output_path_distribution is not None:
        save_file_to_s3(file_path=s3_output_path_distribution, obj=_distributed_values)
        Log().log(msg=f'Save distribution: {s3_output_path_distribution}')
    return [_distributed_values]


if __name__ == '__main__':
    if ARGS.elements:
        ARGS.elements = ast.literal_eval(ARGS.elements[0])
    parallelizer(action=ARGS.action,
                 output_path_distribution=ARGS.output_path_distribution,
                 analytical_data_types_path=ARGS.analytical_data_types_path,
                 data_file_path=ARGS.data_file_path,
                 s3_bucket_name=ARGS.s3_bucket_name,
                 chunks=ARGS.chunks,
                 persist_data=ARGS.persist_data,
                 elements=ARGS.elements,
                 split_by=ARGS.split_by,
                 prefix=ARGS.prefix,
                 sep=ARGS.sep,
                 s3_output_path_distribution=ARGS.s3_output_path_distribution
                 )
