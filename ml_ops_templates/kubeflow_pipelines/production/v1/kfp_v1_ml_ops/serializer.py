"""

Kubeflow Pipeline Component: Serializer

"""

from .container_op_parameters import add_container_op_parameters
from kfp import dsl
from typing import Dict, Union


def serializer(action: str,
               parallelized_obj: Union[list, dsl.PipelineParam],
               aws_account_id: str,
               aws_region: str,
               label_feature_name: str = None,
               labels: list = None,
               output_file_path_missing_data: str = None,
               output_file_path_valid_features: str = None,
               output_file_path_prop_valid_features: str = None,
               output_file_path_n_valid_features: str = None,
               output_file_path_predictors: str = None,
               output_file_path_new_target_feature: str = None,
               s3_output_file_path_parallelized_data: Union[str, dsl.PipelineParam] = None,
               sep: str = ',',
               docker_image_name: str = 'ml-ops-serializer',
               docker_image_tag: str = 'v1',
               volume: dsl.VolumeOp = None,
               volume_dir: str = '/mnt',
               display_name: str = 'Serializer',
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
    Generate container operator for serializer

    :param action: str
        Name of the distribution action
            -> cases: cases to new data set
            -> features: features to new data set
            -> analytical_data_types: distributed analytical data types
            -> processor_memory: distributed processor memory
            -> data_health_check: distributed results of data health check
            -> evolutionary_algorithm: distributed results of ml model optimization

    :param parallelized_obj: list
        List of objects used in parallelization process

    :param aws_account_id: str
        AWS account id

    :param aws_region: str
        AWS region name

    :param label_feature_name: str
            Name of the label feature that contains given labels

    :param labels: list
        Labels used in to identify origin of the cases to serialize

    :param output_file_path_missing_data: str
        Path of the features containing too much missing data output

    :param output_file_path_valid_features: str
        Path of the valid features output

    :param output_file_path_prop_valid_features: str
        Path of the proportion of valid features output

    :param output_file_path_n_valid_features: str
        Path of the number of valid features output

    :param output_file_path_predictors: str
        Path of the predictors output

    :param output_file_path_new_target_feature: str
        Path of the new target feature output

    :param s3_output_file_path_parallelized_data: str
        Complete file path of the parallelized data to save

    :param sep: str
        Separator

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
        Container operator for serializer
    """
    _volume: dict = {volume_dir: volume if volume is None else volume.volume}
    _arguments: list = ['-action', action,
                        '-parallelized_obj', parallelized_obj,
                        '-sep', sep
                        ]
    _file_outputs: Dict[str, str] = {}
    if label_feature_name is not None:
        _arguments.extend(['-label_feature_name', label_feature_name])
    if labels is not None:
        _arguments.extend(['-labels', labels])
    if output_file_path_missing_data is not None:
        _arguments.extend(['-output_file_path_missing_data', output_file_path_missing_data])
        _file_outputs.update({'missing_data': output_file_path_missing_data})
    if output_file_path_valid_features is not None:
        _arguments.extend(['-output_file_path_valid_features', output_file_path_valid_features])
        _file_outputs.update({'valid_features': output_file_path_valid_features})
    if output_file_path_prop_valid_features is not None:
        _arguments.extend(['-output_file_path_prop_valid_features', output_file_path_prop_valid_features])
        _file_outputs.update({'prop_valid_features': output_file_path_prop_valid_features})
    if output_file_path_n_valid_features is not None:
        _arguments.extend(['-output_file_path_n_valid_features', output_file_path_n_valid_features])
        _file_outputs.update({'n_valid_features': output_file_path_n_valid_features})
    if output_file_path_predictors is not None:
        _arguments.extend(['-output_file_path_predictors', output_file_path_predictors])
        _file_outputs.update({'features': output_file_path_predictors})
    if output_file_path_new_target_feature is not None:
        _arguments.extend(['-output_file_path_new_target_feature', output_file_path_new_target_feature])
        _file_outputs.update({'new_target_feature': output_file_path_new_target_feature})
    if s3_output_file_path_parallelized_data is not None:
        _arguments.extend(['-s3_output_file_path_parallelized_data', s3_output_file_path_parallelized_data])
    _task: dsl.ContainerOp = dsl.ContainerOp(name='serializer',
                                             image=f'{aws_account_id}.dkr.ecr.{aws_region}.amazonaws.com/{docker_image_name}:{docker_image_tag}',
                                             command=["python", "task.py"],
                                             arguments=_arguments,
                                             init_containers=None,
                                             sidecars=None,
                                             container_kwargs=None,
                                             artifact_argument_paths=None,
                                             file_outputs=_file_outputs,
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
