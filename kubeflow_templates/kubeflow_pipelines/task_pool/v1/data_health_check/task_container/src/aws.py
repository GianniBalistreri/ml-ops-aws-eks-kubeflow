"""

AWS specific functions

"""

import boto3
import io
import json
import pickle


class AWSS3Exception(Exception):
    """
    Class for handling exceptions for function save_file_to_s3
    """
    pass


def save_file_to_s3(file_path: str, obj) -> None:
    """
    Save file to AWS S3 bucket

    :param file_path: str
        Complete file path

    :param obj:
        File object to save
    """
    _complete_file_path: str = file_path.replace('s3://', '')
    _bucket_name: str = _complete_file_path.split('/')[0]
    _file_path: str = _complete_file_path.replace(_bucket_name, '')
    _file_type: str = _complete_file_path.split('.')[-1]
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
    elif _file_type in ['png', 'jpg', 'jpeg']:
        _buffer: io.BytesIO = io.BytesIO()
        _buffer.write(obj)
        _s3_client.put_object(Body=_buffer.getvalue(), Bucket=_bucket_name, Key=_file_path)
    else:
        raise AWSS3Exception(f'Saving file type ({_file_type}) not supported')
