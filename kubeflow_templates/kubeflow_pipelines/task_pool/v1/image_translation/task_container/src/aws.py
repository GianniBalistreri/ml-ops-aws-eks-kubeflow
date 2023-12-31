"""

AWS specific functions

"""

import boto3
import cv2
import io
import json
import numpy as np
import pickle

from typing import Tuple, Union


class AWSS3Exception(Exception):
    """
    Class for handling exceptions for function save_file_to_s3
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


def load_file_from_s3(file_path: str,
                      image_channels: int = 3,
                      encoding: str = 'utf-8'
                      ) -> Union[dict, object, np.ndarray]:
    """
    Load file from AWS S3 bucket

    :param file_path: str
        Complete file path

    :param image_channels: int
        Number of channels of the image
            -> 1: Grey
            -> 3: Color

    :param encoding: str
        Encoding code

    :return: object
        Loaded file object
    """
    _bucket_name, _file_path, _file_type = _extract_file_path_elements(file_path=file_path)
    _s3_resource: boto3 = boto3.resource('s3')
    if _file_type in ['png', 'jpg', 'jpeg']:
        _image_channels: int = cv2.IMREAD_COLOR if image_channels > 1 else cv2.IMREAD_GREY
        _file_stream: io.BytesIO = io.BytesIO()
        _obj: bytes = _s3_resource.Bucket(_bucket_name).Object(_file_path).download_fileobj(_file_stream)
        _image_array: np.ndarray = np.frombuffer(_file_stream.getbuffer(), dtype="uint8")
        return cv2.imdecode(_image_array, _image_channels)
    else:
        _obj: bytes = _s3_resource.Bucket(_bucket_name).Object(_file_path).get()['Body'].read()
        if _file_type == 'json':
            return json.loads(_obj)
        elif _file_type in ['p', 'pkl', 'pickle']:
            return pickle.loads(_obj)
        elif _file_type == 'txt':
            return _obj.decode(encoding=encoding)
        else:
            raise AWSS3Exception(f'Loading file type ({_file_type}) not supported')


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
    if _file_type in ['png', 'jpg', 'jpeg']:
        _, _image_bytes = cv2.imencode(_file_type, obj)
        _image_bytes = _image_bytes.tobytes()
        _s3_client.put_object(Body=_image_bytes, Bucket=_bucket_name, Key=_file_path, ContentType=f'image/{_file_type}')
    elif _file_type == 'h5':
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
    elif _file_type == 'html':
        _buffer: io.StringIO = io.StringIO()
        _buffer.write(obj.to_html())
        _s3_client.put_object(Body=_buffer.getvalue(), Bucket=_bucket_name, Key=_file_path)
    else:
        raise AWSS3Exception(f'Saving file type ({_file_type}) not supported')
