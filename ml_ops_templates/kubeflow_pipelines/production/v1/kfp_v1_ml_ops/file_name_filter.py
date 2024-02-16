"""

Filter file names by identifier from AWS S3 bucket

"""

from .container_op_parameters import add_container_op_parameters
from kfp import dsl
from kfp.components import create_component_from_func
from typing import List, NamedTuple


class FileNameFilterException(Exception):
    """
    Class for handling exceptions for function file_name_filter
    """
    pass


def _filter(bucket_name: str, obj_ids: str) -> NamedTuple('outputs', [('file_paths', list)]):
    """
    Filter file names by identifier

    :param bucket_name: str
        Name of the target s3 bucket

    :param obj_ids: str
        Identifier to filter by

    :return List[str]
        Filtered complete file names
    """
    import ast
    import boto3
    import os
    _ids: List[str] = ast.literal_eval(obj_ids)
    _file_names: List[str] = []
    _s3_client: boto3 = boto3.client('s3')
    _paginator = _s3_client.get_paginator('list_objects_v2')
    for object_id in _ids:
        for page in _paginator.paginate(Bucket=bucket_name):
            for obj in page.get('Contents', []):
                if object_id in obj['Key']:
                    _file_names.append(os.path.join(f's3://{bucket_name}', obj['Key']))
    return [_file_names]


def file_name_filter(bucket_name: str,
                     obj_ids: List[str],
                     python_version: str = '3.9',
                     display_name: str = 'File Name Filter',
                     n_cpu_request: str = None,
                     n_cpu_limit: str = None,
                     n_gpu: str = None,
                     gpu_vendor: str = 'nvidia',
                     memory_request: str = '100Mi',
                     memory_limit: str = None,
                     ephemeral_storage_request: str = '100Mi',
                     ephemeral_storage_limit: str = None,
                     instance_name: str = 'm5.xlarge',
                     max_cache_staleness: str = 'P0D'
                     ) -> dsl.ContainerOp:
    """
    Filter file names by identifier

    :param bucket_name: str
        Name of the target s3 bucket

    :param obj_ids: List[str]
        Identifier to filter by

    :param python_version: str
        Python version of the base image

    :param display_name: str
        Display name of the Kubeflow Pipeline component

    :param n_cpu_request: str
        Number of requested CPU's

    :param n_cpu_limit: str
        Maximum number of requested CPU's

    :param n_gpu: str
        Maximum number of requested GPU's

    :param gpu_vendor: str
        Name of the GPU vendor
            -> amd: AMD
            -> nvidia: NVIDIA

    :param memory_request: str
        Memory request

    :param memory_limit: str
        Limit of the requested memory

    :param ephemeral_storage_request: str
        Ephemeral storage request (cloud based additional memory storage)

    :param ephemeral_storage_limit: str
        Limit of the requested ephemeral storage (cloud based additional memory storage)

    :param instance_name: str
        Name of the used AWS instance (value)

    :param max_cache_staleness: str
        Maximum of staleness days of the component cache

    :return: dsl.ContainerOp
        Container operator for filter file name filter
    """
    _container_from_func: dsl.component = create_component_from_func(func=_filter,
                                                                     output_component_file=None,
                                                                     base_image=f'python:{python_version}',
                                                                     packages_to_install=['boto3==1.34.11'],
                                                                     annotations=None
                                                                     )
    _task: dsl.ContainerOp = _container_from_func(bucket_name=bucket_name, obj_ids=str(obj_ids))
    _task.set_display_name(display_name)
    add_container_op_parameters(container_op=_task,
                                n_cpu_request=n_cpu_request,
                                n_cpu_limit=n_cpu_limit,
                                n_gpu=n_gpu,
                                gpu_vendor=gpu_vendor,
                                memory_request=memory_request,
                                memory_limit=memory_limit,
                                ephemeral_storage_request=ephemeral_storage_request,
                                ephemeral_storage_limit=ephemeral_storage_limit,
                                instance_name=instance_name,
                                max_cache_staleness=max_cache_staleness
                                )
    return _task
