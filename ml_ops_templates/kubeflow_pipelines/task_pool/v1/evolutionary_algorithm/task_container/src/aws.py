"""

AWS specific functions

"""

import boto3
import io
import json
import pickle

from botocore.errorfactory import ClientError
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
    Check whether file exists in given S3 bucket or not

    :param file_path: str
        Complete file path

    :return: bool
        File exists or not
    """
    try:
        load_file_from_s3(file_path=file_path)
        return True
    except ClientError:
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
    else:
        raise AWSS3Exception(f'Loading file type ({_file_type}) not supported')


def save_file_to_s3(file_path: str, obj) -> None:
    """
    Save file to AWS S3 bucket

    :param file_path: str
        Complete file path

    :param obj:
        File object to save
    """
    _bucket_name, _file_path, _file_type = _extract_file_path_elements(file_path=file_path)
    _s3_resource: boto3 = boto3.resource('s3')
    _s3_obj: _s3_resource.Object = _s3_resource.Object(_bucket_name, _file_path)
    if _file_type == 'json':
        _s3_obj.put(Body=json.dumps(obj=obj))
    elif _file_type in ['p', 'pkl', 'pickle']:
        _s3_obj.put(Body=pickle.dumps(obj=obj, protocol=pickle.HIGHEST_PROTOCOL))
    elif _file_type == 'txt':
        _buffer: io.StringIO = io.StringIO()
        _buffer.write(obj)
        _s3_obj.put_object(Body=_buffer.getvalue(), Bucket=_buffer, Key=_file_path)
    elif _file_type == 'html':
        _buffer: io.StringIO = io.StringIO()
        _buffer.write(obj.to_html())
        _s3_obj.put_object(Body=_buffer.getvalue(), Bucket=_buffer, Key=_file_path)
    else:
        raise AWSS3Exception(f'Saving file type ({_file_type}) not supported')
