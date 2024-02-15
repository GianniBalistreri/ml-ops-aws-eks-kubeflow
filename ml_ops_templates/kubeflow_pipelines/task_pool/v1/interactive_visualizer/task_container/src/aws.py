"""

AWS specific functions

"""

import boto3
import io
import json
import pandas as pd
import pickle

from typing import Tuple, Union


class AWSS3Exception(Exception):
    """
    Class for handling exceptions for module functions
    """
    pass


def _extract_file_path_elements(file_path: str) -> Tuple[str, str, str]:
    """
    Extract file path elements from complete S3 file path

    :param file_path: str
        Complete S3 file path

    :return: Tuple[str, str, str]
        Extracted S3 bucket name, file path and file type
    """
    _complete_file_path: str = file_path.replace('s3://', '')
    _bucket_name: str = _complete_file_path.split('/')[0]
    _file_path: str = _complete_file_path.replace(f'{_bucket_name}/', '')
    _file_type: str = _complete_file_path.split('.')[-1]
    return _bucket_name, _file_path, _file_type


def file_exists(file_path: str) -> bool:
    """
    Check whether file exists in given S3 bucket

    :param file_path: str
        Complete file path

    :return: bool
        File exists or not
    """
    _complete_file_path: str = file_path.replace('s3://', '')
    _bucket_name: str = _complete_file_path.split('/')[0]
    _file_path: str = _complete_file_path.replace(_bucket_name, '')
    _file_type: str = _complete_file_path.split('.')[-1]
    _s3_resource: boto3 = boto3.resource('s3')
    try:
        _s3_resource.head_object(Bucket=_bucket_name, Key=_file_path)
        return True
    except:
        return False


def load_file_from_s3(file_path: str, encoding: str = 'utf-8') -> Union[dict, object]:
    """
    Load file from AWS S3 bucket

    :param file_path: str
        Complete file path

    :param encoding: str
        Encoding code

    :return: object
        Loaded file object
    """
    _bucket_name, _file_path, _file_type = _extract_file_path_elements(file_path=file_path)
    _s3_resource: boto3 = boto3.resource('s3')
    _obj: bytes = _s3_resource.Bucket(_bucket_name).Object(_file_path).get()['Body'].read()
    if _file_type == 'json':
        return json.loads(_obj)
    elif _file_type in ['p', 'pkl', 'pickle']:
        return pickle.loads(_obj)
    elif _file_type == 'txt':
        return _obj.decode(encoding=encoding)
    elif _file_type in ['png', 'jpg', 'jpeg']:
        _file_name: str = _file_path.split('/')[-1]
        _s3_resource.Bucket(_bucket_name).download_file(_file_path, _file_name)
        return _file_name
    else:
        raise AWSS3Exception(f'Loading file type ({_file_type}) not supported')


def load_file_from_s3_as_df(file_path: str, sep: str = ',') -> pd.DataFrame:
    """
    Load file from AWS S3 bucket as Pandas DataFrame

    :param file_path: str
        Complete file path of the data set

    :param sep: str
        Separator value

    :return: pd.DataFrame
        Data set as Pandas DataFrame
    """
    _file_type: str = file_path.split('.')[-1]
    if _file_type == 'json':
        return pd.read_json(path_or_buf=file_path)
    elif _file_type in ['csv', 'txt']:
        return pd.read_csv(filepath_or_buffer=file_path, sep=sep)
    elif _file_type in ['p', 'pkl', 'pickle']:
        return pd.read_pickle(filepath_or_buffer=file_path)
    else:
        raise AWSS3Exception(f'Loading file type ({_file_type}) not supported')


def save_file_to_s3(file_path: str, obj, plotly: bool = True) -> None:
    """
    Save file to AWS S3 bucket

    :param file_path: str
        Complete file path

    :param obj:
        File object to save

    :param plotly: bool
        Whether to save html file from a plot.ly figure or not
    """
    _bucket_name, _file_path, _file_type = _extract_file_path_elements(file_path=file_path)
    _s3_client: boto3.client = boto3.client('s3')
    if _file_type == 'json':
        _s3_client.put_object(Body=json.dumps(obj=obj), Bucket=_bucket_name, Key=_file_path)
    elif _file_type in ['p', 'pkl', 'pickle']:
        _s3_client.put_object(Body=pickle.dumps(obj=obj, protocol=pickle.HIGHEST_PROTOCOL), Bucket=_bucket_name, Key=_file_path)
    elif _file_type == 'txt':
        _buffer: io.StringIO = io.StringIO()
        _buffer.write(obj)
        _s3_client.put_object(Body=_buffer.getvalue(), Bucket=_bucket_name, Key=_file_path)
    elif _file_type == 'html':
        _buffer: io.StringIO = io.StringIO()
        _buffer.write(obj.to_html())
        _s3_client.put_object(Body=_buffer.getvalue(), Bucket=_bucket_name, Key=_file_path)
    elif _file_type in ['jpg', 'jpeg', 'png']:
        if plotly:
            _buffer: io.BytesIO = io.BytesIO()
            obj.write_image(_buffer)
            _buffer.seek(0)
        else:
            _buffer: str = obj
        _s3_client.put_object(Body=_buffer, Bucket=_bucket_name, Key=_file_path)
    else:
        raise AWSS3Exception(f'Saving file type ({_file_type}) not supported')
