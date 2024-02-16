"""

Kubeflow Pipeline Component: Data Typing

"""

from .container_op_parameters import add_container_op_parameters
from kfp import dsl
from typing import Dict, List


def data_typing(data_set_path: str,
                analytical_data_types_path: str,
                s3_output_file_path_data_set: str,
                aws_account_id: str,
                aws_region: str,
                missing_value_features: List[str] = None,
                data_types_config: Dict[str, str] = None,
                sep: str = ',',
                s3_output_file_path_data_typing: str = None,
                docker_image_name: str = 'ml-ops-data-typing',
                docker_image_tag: str = 'v1',
                volume: dsl.VolumeOp = None,
                volume_dir: str = '/mnt',
                display_name: str = 'Data Typing',
                n_cpu_request: str = None,
                n_cpu_limit: str = None,
                n_gpu: str = None,
                gpu_vendor: str = 'nvidia',
                memory_request: str = '1G',
                memory_limit: str = None,
                ephemeral_storage_request: str = '5G',
                ephemeral_storage_limit: str = None,
                instance_name: str = 'm5.xlarge',
                max_cache_staleness: str = 'P0D'
                ) -> dsl.ContainerOp:
    """
    Type features of structured (tabular) data

    :param data_set_path: str
        Complete file path of the data set

    :param analytical_data_types_path: str
        Complete file path of the analytical data types

    :param s3_output_file_path_data_set: str
        Complete file path of the typed data set

    :param aws_account_id: str
        AWS account id

    :param aws_region: str
        AWS region name

    :param missing_value_features: List[str]
        Name of the features containing missing values

    :param data_types_config: Dict[str, str]
        Pre-defined data typing configuration

    :param sep: str
        Separator

    :param s3_output_file_path_data_typing: str
        Complete file path of the data typing output

    :param aws_account_id: str
        AWS account id

    :param docker_image_name: str
        Name of the docker image repository

    :param docker_image_tag: str
        Name of the docker image tag

    :param volume: dsl.VolumeOp
        Attached container volume

    :param volume_dir: str
        Name of the volume directory

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
        Container operator for data health check
    """
    _volume: dict = {volume_dir: volume if volume is None else volume.volume}
    _arguments: list = ['-data_set_path', data_set_path,
                        '-analytical_data_types_path', analytical_data_types_path,
                        '-s3_output_file_path_data_set', s3_output_file_path_data_set,
                        '-sep', sep
                        ]
    if missing_value_features is not None:
        _arguments.extend(['-missing_value_features', missing_value_features])
    if data_types_config is not None:
        _arguments.extend(['-data_types_config', data_types_config])
    if s3_output_file_path_data_typing is not None:
        _arguments.extend(['-s3_output_file_path_data_typing', s3_output_file_path_data_typing])
    _task: dsl.ContainerOp = dsl.ContainerOp(name='data_typing',
                                             image=f'{aws_account_id}.dkr.ecr.{aws_region}.amazonaws.com/{docker_image_name}:{docker_image_tag}',
                                             command=["python", "task.py"],
                                             arguments=_arguments,
                                             init_containers=None,
                                             sidecars=None,
                                             container_kwargs=None,
                                             artifact_argument_paths=None,
                                             file_outputs=None,
                                             output_artifact_paths=None,
                                             is_exit_handler=False,
                                             pvolumes=volume if volume is None else _volume
                                             )
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
