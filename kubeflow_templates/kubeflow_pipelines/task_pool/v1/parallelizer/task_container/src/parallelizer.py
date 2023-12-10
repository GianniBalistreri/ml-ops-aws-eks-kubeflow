"""

Prepare data for processing in parallel

"""

import boto3
import numpy as np
import pandas as pd

from custom_logger import Log
from typing import Dict, List, Union


class ParallelizerException(Exception):
    """
    Class for handling exceptions to module functions
    """
    pass


def distribute_cases(file_path: str, chunks: int, persist_data: bool, sep: str = ',') -> list:
    """
    Distribute cases to different files and container

    :param file_path: str
        Complete path to get file paths to distribute

    :param chunks: int
        Number of chunks to distribute

    :param persist_data: bool
        Whether to persist distributed data in several data set or not

    :param sep: str
        Separator

    :return: list
        Distributed cases
    """
    _df: pd.DataFrame = pd.read_csv(filepath_or_buffer=file_path, sep=sep)
    _pairs_array: List[np.array] = np.array_split(ary=_df.index.values, indices_or_sections=chunks)
    _pairs: list = []
    for i, pair in enumerate(_pairs_array):
        if persist_data:
            _new_file_name: str = file_path.replace('.', f'_{i}.')
            _df_subset: pd.DataFrame = _df.iloc[pair.tolist(), :]
            _df_subset.to_csv(path_or_buf=_new_file_name, sep=sep, index=False)
            _pairs.append(_new_file_name)
        else:
            _pairs.append(pair.tolist())
    Log().log(msg=f'Distributed {_df.shape[0]} cases into {chunks} chunks')
    return _pairs


def distribute_elements(elements: Union[List[str], np.array], chunks: int) -> List[list]:
    """
    Distribute given list elements to different container

    :param elements: Union[List[str], np.array]
        List or array of elements to distribute

    :param chunks: int
        Number of chunks to distribute

    :return: List[list]
        Distributed list elements
    """
    if isinstance(elements, list):
        _array: np.array = np.array(elements)
    else:
        _array: np.array = elements
    _pairs_array: List[np.array] = np.array_split(ary=_array, indices_or_sections=chunks)
    _pairs: list = []
    for pair in _pairs_array:
        _pairs.append(pair.tolist())
    Log().log(msg=f'Distributed {len(elements)} elements into {chunks} chunks')
    return _pairs


def distribute_features(file_path: str, persist_data: bool, chunks: int = None, sep: str = ',') -> list:
    """
    Distribute features to different files and container

    :param file_path: str
        Complete file path of the data set

    :param persist_data: bool
        Whether to persist distributed data in several data set or not

    :param chunks: int
        Number of chunks to distribute

    :param sep: str
        Separator

    :return: list
        Distributed features
    """
    _df: pd.DataFrame = pd.read_csv(filepath_or_buffer=file_path, sep=sep)
    _chunks: int = _df.shape[1] if chunks is None else chunks
    if _chunks > 100:
        _chunks = 100
    _pairs_array: List[np.array] = np.array_split(ary=_df.columns.values, indices_or_sections=_chunks)
    _pairs: list = []
    for i, pair in enumerate(_pairs_array):
        if persist_data:
            _new_file_name: str = file_path.replace('.', f'_{i}.')
            _df_subset: pd.DataFrame = _df.loc[:, pair.tolist()]
            _df_subset.to_csv(path_or_buf=_new_file_name, sep=sep, index=False)
            _pairs.append(_new_file_name)
        else:
            _pairs.append(pair.tolist())
    Log().log(msg=f'Distribute {_df.shape[1]} features into {_chunks} chunks')
    return _pairs


def distribute_file_paths(chunks: int, bucket_name: str, prefix: str = None) -> List[List[str]]:
    """
    Distribute file paths to different container

    :param chunks: int
        Number of chunks to distribute

    :param bucket_name: str
        Name of the S3 bucket

    :param prefix: str
        Prefix to filter by (e.g. dir/file_name)

    :return: List[List[str]]
        Distributed file paths
    """
    _s3_resource: boto3 = boto3.resource('s3')
    _paginator = _s3_resource.get_paginator('list_objects')
    _operation_parameters: Dict[str, str] = {'Bucket': bucket_name}
    if prefix is not None:
        _operation_parameters.update({'Prefix': prefix})
    _page_iterator = _paginator.paginate(**_operation_parameters)
    _file_names: List[str] = []
    for page in _page_iterator:
        for content in page['Contents']:
            _file_names.append(content['Key'])
    _pairs_array: List[np.array] = np.array_split(ary=np.array(_file_names), indices_or_sections=chunks)
    _pairs: List[List[str]] = []
    for pair in _pairs_array:
        _pairs.append(pair.tolist())
    Log().log(msg=f'Distribute {len(_file_names)} files into {chunks} chunks')
    return _pairs
