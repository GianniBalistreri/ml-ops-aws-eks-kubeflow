"""

Kubeflow Pipeline Component: Data Health Check

"""

from .container_op_parameters import add_container_op_parameters
from kfp import dsl
from typing import List


def data_health_check(data_set_path: str,
                      analytical_data_types_path: str,
                      features: List[str] = None,
                      missing_value_threshold: float = 0.95,
                      sep: str = ',',
                      parallel_mode: bool = False,
                      output_file_path_missing_data: str = 'missing_data.json',
                      output_file_path_invariant_features: str = 'invariant_features.json',
                      output_file_path_duplicated_features: str = 'duplicated_features.json',
                      output_file_path_valid_features: str = 'valid_features.json',
                      output_file_path_prop_valid_features: str = 'prop_valid_features.json',
                      output_file_path_n_valid_features: str = 'n_valid_features.json',
                      s3_output_file_path_data_health_check: str = None,
                      aws_account_id: str = '711117404296',
                      docker_image_name: str = 'ml-ops-data-health-check',
                      docker_image_tag: str = 'v1',
                      volume: dsl.VolumeOp = None,
                      volume_dir: str = '/mnt',
                      display_name: str = 'Data Health Check',
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
    Generate container operator for data health check

    :param data_set_path: str
        Complete file path of the data set

    :param analytical_data_types_path: str
        Complete file path of the analytical data types

    :param features: List[str]
        Name of the features to check

    :param output_file_path_missing_data: str
        Path of the features containing too much missing data output

    :param output_file_path_invariant_features: str
        Path of the invariant features

    :param output_file_path_duplicated_features: str
        Path of the duplicated features

    :param output_file_path_valid_features: str
        Path of the valid features output

    :param output_file_path_prop_valid_features: str
        Path of the proportion of valid features output

    :param output_file_path_n_valid_features: str
        Path of the number of valid features output

    :param missing_value_threshold: float
        Threshold of missing values to exclude numeric feature

    :param sep: str
        Separator

    :param parallel_mode: bool
        Whether to run task in parallel mode or not

    :param s3_output_file_path_data_health_check: str
        Complete file path of the data health check output

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
                        '-output_file_path_missing_data', output_file_path_missing_data,
                        '-output_file_path_invariant_features', output_file_path_invariant_features,
                        '-output_file_path_duplicated_features', output_file_path_duplicated_features,
                        '-output_file_path_valid_features', output_file_path_valid_features,
                        '-output_file_path_prop_valid_features', output_file_path_prop_valid_features,
                        '-output_file_path_n_valid_features', output_file_path_n_valid_features,
                        '-missing_value_threshold', missing_value_threshold,
                        '-sep', sep,
                        '-parallel_mode', int(parallel_mode)
                        ]
    if features is not None:
        _arguments.extend(['-features', features])
    if s3_output_file_path_data_health_check is not None:
        _arguments.extend(['-s3_output_file_path_data_health_check', s3_output_file_path_data_health_check])
    _task: dsl.ContainerOp = dsl.ContainerOp(name='data_health_check',
                                             image=f'{aws_account_id}.dkr.ecr.eu-central-1.amazonaws.com/{docker_image_name}:{docker_image_tag}',
                                             command=["python", "task.py"],
                                             arguments=_arguments,
                                             init_containers=None,
                                             sidecars=None,
                                             container_kwargs=None,
                                             artifact_argument_paths=None,
                                             file_outputs={'missing_data': output_file_path_missing_data,
                                                           'invariant_features': output_file_path_invariant_features,
                                                           'duplicated': output_file_path_duplicated_features,
                                                           'valid_features': output_file_path_valid_features,
                                                           'prop_valid_features': output_file_path_prop_valid_features,
                                                           'n_valid_features': output_file_path_n_valid_features,
                                                           },
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
