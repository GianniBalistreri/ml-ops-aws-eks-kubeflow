"""

Kubeflow Pipeline Component: Slack messaging and alerting

"""

from .container_op_parameters import add_container_op_parameters
from kfp import dsl


def slack_alerting(exit_handler: bool,
                   aws_account_id: str,
                   aws_region: str,
                   secret_name_slack: str,
                   header: str = None,
                   msg: str = None,
                   footer: str = None,
                   status: int = None,
                   pipeline_metadata_file_path: str = None,
                   kf_url: str = None,
                   kf_user_namespace: str = None,
                   secret_name_kf: str = None,
                   output_file_path_header: str = 'header.json',
                   output_file_path_message: str = 'message.json',
                   output_file_path_footer: str = 'footer.json',
                   output_file_path_response_status_code: str = 'response_status_code.json',
                   docker_image_name: str = 'ml-ops-slack-alerting',
                   docker_image_tag: str = 'v1',
                   volume: dsl.VolumeOp = None,
                   volume_dir: str = '/mnt',
                   display_name: str = 'Slack Alerting',
                   n_cpu_request: str = None,
                   n_cpu_limit: str = None,
                   n_gpu: str = None,
                   gpu_vendor: str = 'nvidia',
                   memory_request: str = '50Mi',
                   memory_limit: str = None,
                   ephemeral_storage_request: str = '50Mi',
                   ephemeral_storage_limit: str = None,
                   instance_name: str = 'm5.xlarge',
                   max_cache_staleness: str = 'P0D'
                   ) -> dsl.ContainerOp:
    """
    Send messages and alerts to dedicated Slack channel

    :param exit_handler: bool
        Whether task should act like an exit handler

    :param aws_account_id: str
        AWS account id

    :param aws_region: str
        Code of the AWS region

    :param secret_name_slack: str
        Secret name of the secret manager entry containing Slack channel credentials

    :param header: str
        Pre-defined header text

    :param msg: str
        Pre-defined message

    :param footer: str
        Pre-defined footer text

    :param status: int
        Status code of the message:
            -> 0: error
            -> 1: warning
            -> 2: info
            -> 3: succeed

    :param pipeline_metadata_file_path: str
            Complete file path of the pipeline metadata

    :param kf_url: str
            URL of the Kubeflow deployment

    :param kf_user_namespace: str
        Kubeflow namespace

    :param secret_name_kf: str
        Secret name of the secret manager entry containing Kubeflow user credentials

    :param output_file_path_header: str
        File path of the header output

    :param output_file_path_message: str
        File path of the message output

    :param output_file_path_footer: str
        File path of the footer output

    :param output_file_path_response_status_code: str
        File path of the response status code output

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
    _arguments: list = ['-exit_handler', int(exit_handler),
                        '-aws_region', aws_region,
                        '-secret_name_slack', secret_name_slack,
                        '-output_file_path_header', output_file_path_header,
                        '-output_file_path_message', output_file_path_message,
                        '-output_file_path_footer', output_file_path_footer,
                        '-output_file_path_response_status_code', output_file_path_response_status_code
                        ]
    if header is not None:
        _arguments.extend(['-header', header])
    if msg is not None:
        _arguments.extend(['-msg', msg])
    if footer is not None:
        _arguments.extend(['-footer', footer])
    if status is not None:
        _arguments.extend(['-status', status])
    if pipeline_metadata_file_path is not None:
        _arguments.extend(['-pipeline_metadata_file_path', pipeline_metadata_file_path])
    if kf_url is not None:
        _arguments.extend(['-kf_url', kf_url])
    if kf_user_namespace is not None:
        _arguments.extend(['-kf_user_namespace', kf_user_namespace])
    if secret_name_kf is not None:
        _arguments.extend(['-secret_name_kf', secret_name_kf])
    _task: dsl.ContainerOp = dsl.ContainerOp(name='slack_alerting',
                                             image=f'{aws_account_id}.dkr.ecr.{aws_region}.amazonaws.com/{docker_image_name}:{docker_image_tag}',
                                             command=["python", "task.py"],
                                             arguments=_arguments,
                                             init_containers=None,
                                             sidecars=None,
                                             container_kwargs=None,
                                             artifact_argument_paths=None,
                                             file_outputs={'header': output_file_path_header,
                                                           'message': output_file_path_message,
                                                           'footer': output_file_path_footer,
                                                           'response_status_code': output_file_path_response_status_code
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
