"""

Kubeflow Pipeline Component: Parallelizer

"""

from .container_op_parameters import add_container_op_parameters
from kfp import dsl


def parallelizer(action: str,
                 aws_account_id: str,
                 aws_region: str,
                 analytical_data_types_path: str = None,
                 data_file_path: str = None,
                 s3_bucket_name: str = None,
                 chunks: int = 4,
                 persist_data: int = 1,
                 elements: list = None,
                 split_by: str = None,
                 prefix: str = None,
                 sep: str = ',',
                 output_path_distribution: str = 'distribution.json',
                 s3_output_path_distribution: str = None,
                 docker_image_name: str = 'ml-ops-parallelizer',
                 docker_image_tag: str = 'v1',
                 volume: dsl.VolumeOp = None,
                 volume_dir: str = '/mnt',
                 display_name: str = 'Parallelizer',
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

    :param action: str
        Name of the distribution action
            -> cases: cases of given data set
            -> elements: elements of given list
            -> features: features of given data set
            -> file_paths: file path in given S3 bucket

    :param aws_account_id: str
        AWS account id

    :param aws_region: str
        AWS region name

    :param output_path_distribution: str
        Path of the distribution output

    :param analytical_data_types_path: str
        Complete file path of the analytical data types

    :param data_file_path: str
        Complete file path of the data set

    :param s3_bucket_name: str
        Name of the S3 bucket

    :param chunks: int
        Number of chunks to distribute

    :param persist_data: int
        Whether to persist distributed chunks or not

    :param elements: list
        Elements to distribute

    :param split_by: str
            Name of the features to split cases by

    :param prefix: str
        Prefix used for filtering folder in S3 bucket

    :param sep: str
        Separator

    :param s3_output_path_distribution: str
        Complete file path of the distribution output

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
    _arguments: list = ['-action', action,
                        '-output_path_distribution', output_path_distribution,
                        '-chunks', chunks,
                        '-persist_data', persist_data,
                        '-sep', sep
                        ]
    if analytical_data_types_path is not None:
        _arguments.extend(['-analytical_data_types_path', analytical_data_types_path])
    if data_file_path is not None:
        _arguments.extend(['-data_file_path', data_file_path])
    if s3_bucket_name is not None:
        _arguments.extend(['-s3_bucket_name', s3_bucket_name])
    if elements is not None:
        _arguments.extend(['-elements', elements])
    if split_by is not None:
        _arguments.extend(['-split_by', split_by])
    if prefix is not None:
        _arguments.extend(['-prefix', prefix])
    if s3_output_path_distribution is not None:
        _arguments.extend(['-s3_output_path_distribution', s3_output_path_distribution])
    _task: dsl.ContainerOp = dsl.ContainerOp(name='parallelizer',
                                             image=f'{aws_account_id}.dkr.ecr.{aws_region}.amazonaws.com/{docker_image_name}:{docker_image_tag}',
                                             command=["python", "task.py"],
                                             arguments=_arguments,
                                             init_containers=None,
                                             sidecars=None,
                                             container_kwargs=None,
                                             artifact_argument_paths=None,
                                             file_outputs={'distribution': output_path_distribution},
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
