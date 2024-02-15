"""

AWS specific functions

"""

import boto3
import cv2
import io
import json
import numpy as np
import os
import pickle

from typing import List, Tuple, Union


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


def filter_files_from_s3(file_path: str, obj_ids: List[str]) -> List[str]:
    """
    Filter files containing certain characteristics from AWS S3 bucket

    :return: List[str]
        Complete file paths
    """
    _complete_file_path: str = file_path.replace('s3://', '')
    _bucket_name: str = _complete_file_path.split('/')[0]
    _file_names: List[str] = []
    _s3_client: boto3 = boto3.client('s3')
    _paginator = _s3_client.get_paginator('list_objects_v2')
    for object_id in obj_ids:
        for page in _paginator.paginate(Bucket=_bucket_name):
            for obj in page.get('Contents', []):
                if object_id in obj['Key']:
                    _file_names.append(os.path.join(f's3://{_bucket_name}', obj['Key']))
    return _file_names


def load_file_from_s3(file_path: str, local_file_path: str = None, image_channels: int = 3) -> Union[None, np.ndarray]:
    """
    Load file from AWS S3 bucket

    :param file_path: str
        Complete file path

    :param local_file_path: str
        File path of the locally saved file

    :param image_channels: int
        Number of channels of the image
            -> 1: Grey
            -> 3: Color

    :return: object
        Loaded image array or None
    """
    _bucket_name, _file_path, _file_type = _extract_file_path_elements(file_path=file_path)
    _s3_resource: boto3 = boto3.resource('s3')
    if local_file_path is None:
        if _file_type in ['jpg', 'jpeg', 'png']:
            _image_channels: int = 3 if image_channels > 1 else 1
            _file_stream: io.BytesIO = io.BytesIO()
            _obj: bytes = _s3_resource.Bucket(_bucket_name).Object(_file_path).download_fileobj(_file_stream)
            _image_array: np.ndarray = np.frombuffer(_file_stream.getbuffer(), dtype="uint8")
            return cv2.imdecode(_image_array, _image_channels)
        else:
            raise AWSS3Exception(f'Loading file type ({_file_type}) not supported')
    else:
        _s3_resource.meta.client.download_file(_bucket_name, _file_path, local_file_path)


def save_file_to_s3(file_path: str, obj, input_file_path: str = None) -> None:
    """
    Save file to AWS S3 bucket

    :param file_path: str
        Complete file path

    :param obj:
        File object to save

    :param input_file_path: str
        Complete input file path of the file to upload to S3
    """
    _bucket_name, _file_path, _file_type = _extract_file_path_elements(file_path=file_path)
    _s3_client: boto3.client = boto3.client('s3')
    if _file_type == 'h5':
        _s3_client.upload_file(Filename=input_file_path,
                               Bucket=_bucket_name,
                               Key=_file_path
                               )
    elif _file_type == 'json':
        _s3_client.put_object(Body=json.dumps(obj=obj), Bucket=_bucket_name, Key=_file_path)
    elif _file_type in ['p', 'pkl', 'pickle']:
        _s3_client.put_object(Body=pickle.dumps(obj=obj, protocol=pickle.HIGHEST_PROTOCOL), Bucket=_bucket_name, Key=_file_path)
    elif _file_type == 'txt':
        _buffer: io.StringIO = io.StringIO()
        _buffer.write(obj)
        _s3_client.put_object(Body=_buffer.getvalue(), Bucket=_bucket_name, Key=_file_path)
    else:
        raise AWSS3Exception(f'Saving file type ({_file_type}) not supported')
