"""

Kubeflow Pipeline Component: Imputation

"""

from .container_op_parameters import add_container_op_parameters
from kfp import dsl
from typing import List, Union


def imputation(data_set_path: str,
               features: List[str],
               s3_output_path_imputed_data_set: str,
               aws_account_id: str,
               aws_region: str,
               imp_meth: str = 'multiple',
               multiple_meth: str = 'random',
               single_meth: str = 'constant',
               constant_value: Union[int, float] = None,
               m: int = 3,
               convergence_threshold: float = 0.99,
               mice_config: dict = None,
               imp_config: dict = None,
               analytical_data_types_path: str = None,
               sep: str = ',',
               output_file_path_imp_features: str = 'imputed_features.json',
               docker_image_name: str = 'ml-ops-imputation',
               docker_image_tag: str = 'v1',
               volume: dsl.VolumeOp = None,
               volume_dir: str = '/mnt',
               display_name: str = 'Imputation',
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
    Generate container operator for missing value imputation

    :param data_set_path: str
        Complete file path of the data set

    :param features: List[str]
        Name of the features

    :param s3_output_path_imputed_data_set: str
        Complete file path of the imputed data set

    :param aws_account_id: str
        AWS account id

    :param aws_region: str
        AWS region name

    :param imp_meth: str
            Name of the imputation method
                -> single: Single imputation
                -> multiple: Multiple Imputation

    :param multiple_meth: str
        Name of the multiple imputation method
            -> mice: Multiple Imputation by Chained Equation
            -> random: Random

    :param single_meth: str
        Name of the single imputation method
            -> constant: Constant value
            -> min: Minimum observed value
            -> max: Maximum observed value
            -> median: Median of observed values
            -> mean: Mean of observed values

    :param constant_value: Union[int, float]
        Constant imputation value used for single imputation method constant

    :param m: int
        Number of chains (multiple imputation)

    :param convergence_threshold: float
        Convergence threshold used for multiple imputation

    :param mice_config: dict
        -

    :param imp_config: dict
        Assignment of different imputation methods to features
            -> key: feature name
            -> value: tuple(imp_meth, meth, constant_value)

    :param analytical_data_types_path: str
        Complete file path of the analytical data types

    :param sep: str
        Separator

    :param output_file_path_imp_features: str
        File path of the imputed feature names

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
        Container operator for missing value imputation
    """
    _volume: dict = {volume_dir: volume if volume is None else volume.volume}
    _arguments: list = ['-data_set_path', data_set_path,
                        '-features', features,
                        '-s3_output_path_imputed_data_set', s3_output_path_imputed_data_set,
                        '-output_file_path_imp_features', output_file_path_imp_features,
                        '-imp_meth', imp_meth,
                        '-multiple_meth', multiple_meth,
                        '-single_meth', single_meth,
                        '-m', m,
                        '-convergence_threshold', convergence_threshold,
                        '-sep', sep
                        ]
    if constant_value is not None:
        _arguments.extend(['-constant_value', constant_value])
    if mice_config is not None:
        _arguments.extend(['-mice_config', mice_config])
    if imp_config is not None:
        _arguments.extend(['-imp_config', imp_config])
    if analytical_data_types_path is not None:
        _arguments.extend(['-analytical_data_types_path', analytical_data_types_path])
    _task: dsl.ContainerOp = dsl.ContainerOp(name='imputation',
                                             image=f'{aws_account_id}.dkr.ecr.{aws_region}.amazonaws.com/{docker_image_name}:{docker_image_tag}',
                                             command=["python", "task.py"],
                                             arguments=_arguments,
                                             init_containers=None,
                                             sidecars=None,
                                             container_kwargs=None,
                                             artifact_argument_paths=None,
                                             file_outputs={'features': output_file_path_imp_features},
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
