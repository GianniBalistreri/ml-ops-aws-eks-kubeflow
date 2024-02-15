"""
Kubeflow Pipeline: kfp v1
"""

import kfp
import os

from container_op_parameters import add_container_op_parameters
from get_session_cookie_dex import get_istio_auth_session
from kfp import dsl
from kfp.compiler import compiler
from typing import NamedTuple


def for_loop(start: int, end: int) -> NamedTuple('outputs', [('range', list)]):
    """
    Generate number of containers running in parallel

    :param start: int
        Start value

    :param end: int
        End value

    :return: NamedTuple
        Output values
    """
    _range: list = [i for i in range(start, end, 1)]
    return [_range]


def task_1(a: float, b: float, volume: dsl.VolumeOp = None, volume_dir: str = '/mnt') -> dsl.ContainerOp:
    """
    Generate container operator for task 1

    :param a: float
        Numeric value

    :param b: float
        Numeric value

    :param volume: dsl.VolumeOp
        Attached container volume

    :param volume_dir: str
        Name of the volume directory

    :return: dsl.ContainerOp
        Container operator for task 1
    """
    _volume: dict = {volume_dir: volume.volume}
    _task: dsl.ContainerOp = dsl.ContainerOp(name='xxx',
                                             image='xxx',
                                             command=None,
                                             arguments=['--a', a,
                                                        '--b', b
                                                        ],
                                             init_containers=None,
                                             sidecars=None,
                                             container_kwargs=None,
                                             artifact_argument_paths=None,
                                             file_outputs={'output': '/task_1_output.txt'},
                                             output_artifact_paths=None,
                                             is_exit_handler=False,
                                             pvolumes=volume if volume is None else _volume
                                             )
    _task.set_display_name('Task 1')
    _task = add_container_op_parameters(container_op=_task,
                                        n_cpu_request='1',
                                        n_cpu_limit=None,
                                        n_gpu=None,
                                        memory_request='1G',
                                        memory_limit=None,
                                        ephemeral_storage_request='5G',
                                        ephemeral_storage_limit=None,
                                        instance_name='m5.xlarge'
                                        )
    return _task


def task_2(a: float, b: float, volume: dsl.VolumeOp = None, volume_dir: str = '/mnt') -> dsl.ContainerOp:
    """
    Generate container operator for task 2

    :param a: float
        Numeric value

    :param b: float
        Numeric value

    :param volume: dsl.VolumeOp
        Attached container volume

    :param volume_dir: str
        Name of the volume directory

    :return: dsl.ContainerOp
        Container operator for task 2
    """
    _volume: dict = {volume_dir: volume}
    _task: dsl.ContainerOp = dsl.ContainerOp(name='xxx',
                                             image='xxx',
                                             command=None,
                                             arguments=['--a', a,
                                                        '--b', b
                                                        ],
                                             init_containers=None,
                                             sidecars=None,
                                             container_kwargs=None,
                                             artifact_argument_paths=None,
                                             file_outputs={'output': '/task_2_output.txt'},
                                             output_artifact_paths=None,
                                             is_exit_handler=False,
                                             pvolumes=volume if volume is None else _volume
                                             )
    _task.set_display_name('Task 2')
    _task = add_container_op_parameters(container_op=_task,
                                        n_cpu_request='1',
                                        n_cpu_limit=None,
                                        n_gpu=None,
                                        memory_request='1G',
                                        memory_limit=None,
                                        ephemeral_storage_request='5G',
                                        ephemeral_storage_limit=None,
                                        instance_name='m5.xlarge'
                                        )
    return _task


def task_3(a: float, b: float, volume: dsl.VolumeOp = None, volume_dir: str = '/mnt') -> dsl.ContainerOp:
    """
    Generate container operator for task 3

    :param a: float
        Numeric value

    :param b: float
        Numeric value

    :param volume: dsl.VolumeOp
        Attached container volume

    :param volume_dir: str
        Name of the volume directory

    :return: dsl.ContainerOp
        Container operator for task 3
    """
    _volume: dict = {volume_dir: volume}
    _task: dsl.ContainerOp = dsl.ContainerOp(name='xxx',
                                             image='xxx',
                                             command=None,
                                             arguments=['--a', a,
                                                        '--b', b
                                                        ],
                                             init_containers=None,
                                             sidecars=None,
                                             container_kwargs=None,
                                             artifact_argument_paths=None,
                                             file_outputs={'output': '/task_3_output.txt'},
                                             output_artifact_paths=None,
                                             is_exit_handler=False,
                                             pvolumes=volume if volume is None else _volume
                                             )
    _task.set_display_name('Task 3')
    _task = add_container_op_parameters(container_op=_task,
                                        n_cpu_request='1',
                                        n_cpu_limit=None,
                                        n_gpu=None,
                                        memory_request='1G',
                                        memory_limit=None,
                                        ephemeral_storage_request='5G',
                                        ephemeral_storage_limit=None,
                                        instance_name='m5.xlarge'
                                        )
    return _task


@dsl.pipeline(
    name='xxx',
    description='xxx'
)
def pipeline() -> None:
    """
    Generate Kubeflow Pipeline (v1)
    """
    _volume: dsl.VolumeOp = dsl.VolumeOp(resource_name='xxx',
                                         size='1Gi',
                                         storage_class=None,
                                         modes=None,
                                         annotations=None,
                                         data_source=None,
                                         volume_name=None,
                                         generate_unique_name=True
                                         )
    _task_1: dsl.ContainerOp = task_1(a=1, b=2, volume=_volume)
    _task_2: dsl.ContainerOp = task_2(a=1, b=_task_1.output, volume=_task_1.pvolume)
    with dsl.Condition(condition=_task_2.output < 4, name='Condition 1'):
        _task_3: dsl.ContainerOp = task_3(a=1, b=_task_2.output, volume=_task_2.pvolume)
    with dsl.Condition(condition=_task_2.output >= 4, name='Condition 2'):
        _task_3: dsl.ContainerOp = task_3(a=_task_2.output, b=2, volume=_task_2.pvolume)
    _for_loop = kfp.components.create_component_from_func(func=for_loop,
                                                          output_component_file=None,
                                                          base_image='python:3.9',
                                                          packages_to_install=None,
                                                          annotations=None
                                                          )
    _task_for_loop = _for_loop(start=1, end=3)
    with dsl.ParallelFor(loop_args=_task_for_loop.output, parallelism=None) as item:
        _task_4: dsl.ContainerOp = task_1(a=1, b=4, volume=_task_3.pvolume)
    _task_5: dsl.ContainerOp = task_1(a=2, b=6, volume=_task_3.pvolume).after(_task_4)
    _task_8: dsl.ContainerOp = task_1(a=_task_5.output, b=4, volume=_task_5.pvolume)
    with dsl.ExitHandler(exit_op=_task_8, name='Exit 1'):
        _task_6: dsl.ContainerOp = task_2(a=_task_5.output, b=_task_5.output, volume=_task_5.pvolume)
        _task_7: dsl.ContainerOp = task_2(a=_task_6.output, b=_task_5.output, volume=_task_6.pvolume)


if __name__ == '__main__':
    _auth_cookies: str = get_istio_auth_session(url=os.getenv('KF_URL'),
                                                username=os.getenv('USER_NAME'),
                                                password=os.getenv('PWD')
                                                )["session_cookie"]
    _kfp_client = kfp.Client(host=f"{os.getenv('KF_URL')}/pipeline",
                             client_id=None,
                             namespace=os.getenv('KF_USER_NAMESPACE'),
                             other_client_id=None,
                             other_client_secret=None,
                             existing_token=None,
                             cookies=_auth_cookies,
                             proxy=None,
                             ssl_ca_cert=None,
                             kube_context=None,
                             credentials=None
                             )
    compiler.Compiler().compile(pipeline_func=pipeline,
                                package_path=f"{os.getenv('PIPELINE_NAME')}.yaml",
                                type_check=True,
                                pipeline_conf=None
                                )
    _recurring: bool = os.getenv('RECURRING', default=True)
    if _recurring:
        _kfp_client.create_experiment(name='xxx',
                                      description='xxx',
                                      namespace=os.getenv('KF_USER_NAMESPACE')
                                      )
        _experiment_id: str = _kfp_client.get_experiment(experiment_id=None,
                                                         experiment_name='xxx',
                                                         namespace=os.getenv('KF_USER_NAMESPACE')
                                                         ).id
        _kfp_client.create_recurring_run(experiment_id=_experiment_id,
                                         job_name='xxx',
                                         description='xxx',
                                         start_time=os.getenv('START_TIME'),
                                         end_time=os.getenv('END_TIME'),
                                         interval_second=os.getenv('INTERVAL_SECOND'),
                                         cron_expression=os.getenv('CRON_EXPRESSION', default='0 0 0 * * *'),
                                         max_concurrency=1,
                                         no_catchup=os.getenv('NO_CATCHUP', default=True),
                                         params=None,
                                         pipeline_package_path='xxx.yaml',
                                         pipeline_id=None,
                                         version_id=None,
                                         enabled=True,
                                         enable_caching=True,
                                         service_account=None
                                         )
    else:
        _kfp_client.create_run_from_pipeline_func(pipeline_func=pipeline,
                                                  arguments={},
                                                  run_name='xxx',
                                                  experiment_name='xxx',
                                                  namespace=os.getenv('KF_USER_NAMESPACE'),
                                                  pipeline_root=None,
                                                  enable_caching=True,
                                                  service_account=None
                                                  )
