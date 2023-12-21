"""

Prepare data for processing in parallel

"""

import pandas as pd

from aws import load_file_from_s3
from custom_logger import Log
from typing import List


class SerializerException(Exception):
    """
    Class for handling exceptions to module functions
    """
    pass


def serialize_cases(file_paths: List[str], output_file_path: str, sep: str = ',') -> None:
    """
    Serialize cases of different files and container

    :param file_paths: List[str]
        Complete file paths to serialize

    :param output_file_path: str
        Complete file path of the output data set

    :param sep: str
        Separator
    """
    _df: pd.DataFrame = pd.DataFrame()
    for file_path in file_paths:
        _df_chunk: pd.DataFrame = pd.read_csv(filepath_or_buffer=file_path, sep=sep)
        _df = pd.concat(objs=[_df, _df_chunk], axis=0)
    _df.to_csv(path_or_buf=output_file_path, sep=sep, header=True, index=False)
    Log().log(msg=f'Serialize {_df.shape[0]} cases from {len(file_paths)} chunks')


def serialize_features(file_paths: List[str], output_file_path: str, sep: str = ',') -> None:
    """
    Serialize features of different files and container

    :param file_paths: List[str]
        Complete file paths to serialize

    :param output_file_path: str
        Complete file path of the output data set

    :param sep: str
        Separator
    """
    _df: pd.DataFrame = pd.DataFrame()
    for file_path in file_paths:
        _df_chunk: pd.DataFrame = pd.read_csv(filepath_or_buffer=file_path, sep=sep)
        _df = pd.concat(objs=[_df, _df_chunk], axis=1)
    _df.to_csv(path_or_buf=output_file_path, sep=sep, header=True, index=False)
    Log().log(msg=f'Serialize {_df.shape[0]} features from {len(file_paths)} chunks')


def serialize_evolutionary_results(contents: List[dict]) -> dict:
    """
    Serialize json file from different json file

    :param contents: List[dict]
        List of evolutionary algorithm parallelization contents

    :return: dict
        Serialized json file content
    """
    _serialized_contents: dict = {}
    for content in contents:
        _key: str = list(content.keys())[0]
        _param_file_path: str = content[_key]['param_file_path']
        _eval_metric_file_path: str = content[_key]['eval_metric_file_path']
        _param: dict = load_file_from_s3(file_path=_param_file_path)
        _eval_metric: dict = load_file_from_s3(file_path=_eval_metric_file_path)
        _serialized_contents.update({_key: content[_key]})
        _serialized_contents[_key].update({'parameter': _param, 'metric': _eval_metric})
    Log().log(msg=f'Serialize dictionary from {len(contents)} chunks')
    return _serialized_contents
