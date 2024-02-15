"""

Kubeflow Pipeline Component: Analytical Data Types

"""

from .container_op_parameters import add_container_op_parameters
from kfp import dsl
from typing import List


def analytical_data_types(data_set_path: str,
                          s3_output_file_path_analytical_data_types,
                          max_categories: int = 100,
                          date_edges: dict = None,
                          categorical: List[str] = None,
                          ordinal: List[str] = None,
                          continuous: List[str] = None,
                          date: List[str] = None,
                          id_text: List[str] = None,
                          sep: str = ',',
                          output_file_path_analytical_data_types: str = 'analytical_data_types.json',
                          output_file_path_categorical_features: str = 'categorical_features.json',
                          output_file_path_ordinal_features: str = 'ordinal_features.json',
                          output_file_path_continuous_features: str = 'continuous_features.json',
                          output_file_path_date_features: str = 'date_features.json',
                          output_file_path_id_text_features: str = 'id_text_features.json',
                          aws_account_id: str = '711117404296',
                          docker_image_name: str = 'ml-ops-analytical-data-types',
                          docker_image_tag: str = 'v1',
                          volume: dsl.VolumeOp = None,
                          volume_dir: str = '/mnt',
                          display_name: str = 'Analytical Data Types',
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
    Generate container operator for analytical data types

    :param data_set_path: str
        Complete file path of the data set

    :param output_file_path_analytical_data_types: str
        File path of the analytical data types output

    :param output_file_path_categorical_features: str
        File path of the categorical features output

    :param output_file_path_ordinal_features: str
        File path of the ordinal features output

    :param output_file_path_continuous_features: str
        File path of the continuous features output

    :param output_file_path_date_features: str
        File path of the date features output

    :param output_file_path_id_text_features: str
        File path of the id / text features output

    :param s3_output_file_path_analytical_data_types: str
        Complete file path of the analytical data types output

    :param max_categories: int
        Maximum number of categories for identifying feature as categorical

    :param date_edges: Tuple[str, str]
            Date boundaries to identify datetime features

    :param categorical: List[str]
            Pre-assigned categorical features

    :param ordinal: List[str]
        Pre-assigned ordinal features

    :param continuous: List[str]
        Pre-assigned continuous features

    :param date: List[str]
        Pre-assigned date features

    :param id_text: List[str]
        Pre-assigned id_text features

    :param sep: str
        Separator

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
        Container operator for analytical data types
    """
    _volume: dict = {volume_dir: volume if volume is None else volume.volume}
    _arguments: list = ['-data_set_path', data_set_path,
                        '-output_file_path_analytical_data_types', output_file_path_analytical_data_types,
                        '-output_file_path_categorical_features', output_file_path_categorical_features,
                        '-output_file_path_ordinal_features', output_file_path_ordinal_features,
                        '-output_file_path_continuous_features', output_file_path_continuous_features,
                        '-output_file_path_date_features', output_file_path_date_features,
                        '-output_file_path_id_text_features', output_file_path_id_text_features,
                        '-s3_output_file_path_analytical_data_types', s3_output_file_path_analytical_data_types,
                        '-max_categories', max_categories,
                        '-sep', sep,
                        ]
    if date_edges is not None:
        _arguments.extend(['-date_edges', date_edges])
    if categorical is not None:
        _arguments.extend(['-categorical', categorical])
    if ordinal is not None:
        _arguments.extend(['-ordinal', ordinal])
    if continuous is not None:
        _arguments.extend(['-continuous', continuous])
    if date is not None:
        _arguments.extend(['-date', date])
    if id_text is not None:
        _arguments.extend(['-id_text', id_text])
    _task: dsl.ContainerOp = dsl.ContainerOp(name='analytical_data_types',
                                             image=f'{aws_account_id}.dkr.ecr.eu-central-1.amazonaws.com/{docker_image_name}:{docker_image_tag}',
                                             command=["python", "task.py"],
                                             arguments=_arguments,
                                             init_containers=None,
                                             sidecars=None,
                                             container_kwargs=None,
                                             artifact_argument_paths=None,
                                             file_outputs={'analytical_data_types': output_file_path_analytical_data_types,
                                                           'categorical_features': output_file_path_categorical_features,
                                                           'ordinal_features': output_file_path_ordinal_features,
                                                           'continuous_features': output_file_path_continuous_features,
                                                           'date_features': output_file_path_date_features,
                                                           'id_text_features': output_file_path_id_text_features
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
