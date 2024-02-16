"""

Kubeflow Pipeline Component: Sampling

"""

from .container_op_parameters import add_container_op_parameters
from kfp import dsl
from typing import Dict, List


def sampling(action: str,
             data_set_file_path: str,
             target_feature: str,
             aws_account_id: str,
             aws_region: str,
             output_file_path_sampling_metadata: str = 'metadata.json',
             output_file_path_sampling_file_paths: str = 'file_paths.json',
             s3_output_file_path_train_data_set: str = None,
             s3_output_file_path_test_data_set: str = None,
             s3_output_file_path_val_data_set: str = None,
             s3_output_file_path_sampling_data_set: str = None,
             features: List[str] = None,
             time_series_feature: str = None,
             train_size: float = 0.8,
             validation_size: float = 0.1,
             random_sample: bool = True,
             target_class_value: int = None,
             target_proportion: float = None,
             size: int = None,
             prop: float = None,
             quotas: Dict[str, Dict[str, float]] = None,
             sep: str = ',',
             s3_output_file_path_sampling_metadata: str = None,
             docker_image_name: str = 'ml-ops-sampling',
             docker_image_tag: str = 'v1',
             volume: dsl.VolumeOp = None,
             volume_dir: str = '/mnt',
             display_name: str = 'Sampling',
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
    Sampling data sets for training, testing and validation used for applying supervised machine learning models

    :param action: str
        Name of the sampling action
            -> random: Random sampling
            -> quota: Quota based sampling
            -> down: Down-sampling of class value
            -> up: Up-sampling of class value
            -> train_test: Train-test sampling for structured data
            -> train_test_time_series: Train-test sampling for time series data

    :param data_set_file_path: str
        Complete file path of the data set

    :param target_feature: str
        Name of the target feature

    :param aws_account_id: str
        AWS account id

    :param aws_region: str
        AWS region name

    :param output_file_path_sampling_metadata: str
        Complete file path of the sampling metadata output

    :param output_file_path_sampling_file_paths: str
        File path of the file sampling output

    :param s3_output_file_path_train_data_set: str
        Complete file path of the sampled training data set

    :param s3_output_file_path_test_data_set: str
        Complete file path of the sampled test data set

    :param s3_output_file_path_val_data_set: str
        Complete file path of the sampled validation data set

    :param s3_output_file_path_sampling_data_set: str
        Complete file path of the sampled data set

    :param features: List[str]
        Name of features to use

    :param time_series_feature: str
        Name of the datetime feature to use

    :param train_size: float
        Size of the training data set

    :param validation_size: float
        Size of the validation data set

    :param random_sample: bool
        Whether to sample randomly or not

    :param target_class_value: Union[str, int]
        Class value of the target feature to sample

    :param target_proportion: float
        Target proportion of the class value of the target feature

    :param size: int
        Sample size

    :param prop: float
        Proportion of the sample size

    :param quotas: Dict[str, Dict[str, float]]
        Pre-defined quota config used for quota sampling

    :param sep: str
        Separator

    :param s3_output_file_path_sampling_metadata: str
        Complete file path of the sampling metadata

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
        Container operator for sampling
    """
    _volume: dict = {volume_dir: volume if volume is None else volume.volume}
    _arguments: list = ['-action', action,
                        '-data_set_file_path', data_set_file_path,
                        '-target_feature', target_feature,
                        '-train_size', train_size,
                        '-validation_size', validation_size,
                        '-random_sample', int(random_sample),
                        '-sep', sep,
                        '-output_file_path_sampling_metadata', output_file_path_sampling_metadata,
                        '-output_file_path_sampling_file_paths', output_file_path_sampling_file_paths
                        ]
    if s3_output_file_path_train_data_set is not None:
        _arguments.extend(['-s3_output_file_path_train_data_set', s3_output_file_path_train_data_set])
    if s3_output_file_path_test_data_set is not None:
        _arguments.extend(['-s3_output_file_path_test_data_set', s3_output_file_path_test_data_set])
    if s3_output_file_path_val_data_set is not None:
        _arguments.extend(['-s3_output_file_path_val_data_set', s3_output_file_path_val_data_set])
    if s3_output_file_path_sampling_data_set is not None:
        _arguments.extend(['-s3_output_file_path_sampling_data_set', s3_output_file_path_sampling_data_set])
    if features is not None:
        _arguments.extend(['-features', features])
    if time_series_feature is not None:
        _arguments.extend(['-time_series_feature', time_series_feature])
    if target_class_value is not None:
        _arguments.extend(['-target_class_value', target_class_value])
    if target_proportion is not None:
        _arguments.extend(['-target_proportion', target_proportion])
    if size is not None:
        _arguments.extend(['-size', size])
    if prop is not None:
        _arguments.extend(['-prop', prop])
    if quotas is not None:
        _arguments.extend(['-quotas', quotas])
    if s3_output_file_path_sampling_metadata is not None:
        _arguments.extend(['-s3_output_file_path_sampling_metadata', s3_output_file_path_sampling_metadata])
    _task: dsl.ContainerOp = dsl.ContainerOp(name='sampling',
                                             image=f'{aws_account_id}.dkr.ecr.{aws_region}.amazonaws.com/{docker_image_name}:{docker_image_tag}',
                                             command=["python", "task.py"],
                                             arguments=_arguments,
                                             init_containers=None,
                                             sidecars=None,
                                             container_kwargs=None,
                                             artifact_argument_paths=None,
                                             file_outputs={'metadata': output_file_path_sampling_metadata,
                                                           'file_paths': output_file_path_sampling_file_paths
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
