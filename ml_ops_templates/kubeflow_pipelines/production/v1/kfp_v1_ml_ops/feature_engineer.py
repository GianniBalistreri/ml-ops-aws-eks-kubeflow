"""

Kubeflow Pipeline Component: Feature Engineer

"""

from .container_op_parameters import add_container_op_parameters
from kfp import dsl
from typing import Any, List


def feature_engineer(data_set_path: Any,
                     analytical_data_types_path: str,
                     target_feature: str,
                     s3_output_file_path_data_set: Any,
                     s3_output_file_path_processor_memory: str,
                     re_engineering: bool = False,
                     next_level: bool = False,
                     feature_engineering_config: str = None,
                     features: List[str] = None,
                     ignore_features: List[str] = None,
                     exclude_features: List[str] = None,
                     exclude_original_data: bool = False,
                     exclude_meth: List[str] = None,
                     use_only_meth: List[str] = None,
                     sep: str = ',',
                     parallel_mode: bool = False,
                     output_file_path_predictors: str = 'features.json',
                     output_file_path_new_target_feature: str = 'new_target_feature.json',
                     aws_account_id: str = '711117404296',
                     docker_image_name: str = 'ml-ops-feature-engineering',
                     docker_image_tag: str = 'v1',
                     volume: dsl.VolumeOp = None,
                     volume_dir: str = '/mnt',
                     display_name: str = 'Feature Engineer',
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
    Generate container operator for feature engineering

    :param data_set_path: str
        Complete file path of the data set

    :param analytical_data_types_path: str
        Complete file path of the analytical data types

    :param target_feature: str
        Name of the target feature

    :param s3_output_file_path_data_set: str
        Complete file path of the data set to save

    :param s3_output_file_path_processor_memory: str
        Complete file path of the processing memory to save

    :param output_file_path_predictors: str
        Path of the predictors output

    :param output_file_path_new_target_feature: str
        Path of the new target feature output

    :param re_engineering: bool
        Whether to re-engineer features for inference or to engineer for training

    :param next_level: bool
        Whether to engineer deeper (higher level) features or first level features

    :param feature_engineering_config: str
            Pre-defined configuration

    :param features: List[str]
        Name of the features

    :param ignore_features: List[str]
        Name of the features to ignore in feature engineering

    :param exclude_features: List[str]
        Name of the features to exclude

    :param exclude_original_data: bool
        Exclude original features

    :param exclude_meth: List[str]
        Name of the feature engineering methods to exclude

    :param use_only_meth: List[str]
        Name of the feature engineering methods to use only

    :param sep: str
        Separator

    :param parallel_mode: bool
        Whether to run task in parallel mode or not

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
        Container operator for feature engineering
    """
    _volume: dict = {volume_dir: volume if volume is None else volume.volume}
    _arguments: list = ['-data_set_path', data_set_path,
                        '-analytical_data_types_path', analytical_data_types_path,
                        '-target_feature', target_feature,
                        '-s3_output_file_path_data_set', s3_output_file_path_data_set,
                        '-s3_output_file_path_processor_memory', s3_output_file_path_processor_memory,
                        '-output_file_path_predictors', output_file_path_predictors,
                        '-output_file_path_new_target_feature', output_file_path_new_target_feature,
                        '-re_engineering', int(re_engineering),
                        '-next_level', int(next_level),
                        '-exclude_original_data', int(exclude_original_data),
                        '-sep', sep,
                        '-parallel_mode', int(parallel_mode)
                        ]
    if feature_engineering_config is not None:
        _arguments.extend(['-feature_engineering_config', feature_engineering_config])
    if features is not None:
        _arguments.extend(['-features', features])
    if ignore_features is not None:
        _arguments.extend(['-ignore_features', ignore_features])
    if exclude_features is not None:
        _arguments.extend(['-exclude_features', exclude_features])
    if exclude_meth is not None:
        _arguments.extend(['-exclude_meth', exclude_meth])
    if use_only_meth is not None:
        _arguments.extend(['-use_only_meth', use_only_meth])
    _task: dsl.ContainerOp = dsl.ContainerOp(name='feature_engineer',
                                             image=f'{aws_account_id}.dkr.ecr.eu-central-1.amazonaws.com/{docker_image_name}:{docker_image_tag}',
                                             command=["python", "task.py"],
                                             arguments=_arguments,
                                             init_containers=None,
                                             sidecars=None,
                                             container_kwargs=None,
                                             artifact_argument_paths=None,
                                             file_outputs={'features': output_file_path_predictors,
                                                           'new_target_feature': output_file_path_new_target_feature
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
