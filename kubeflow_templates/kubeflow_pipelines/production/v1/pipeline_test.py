"""
Kubeflow Pipeline: kfp v1
"""

import kfp
import os

from container_op_parameters import add_container_op_parameters
from get_session_cookie_dex import get_istio_auth_session
from kfp import dsl
from kfp.compiler import compiler


def task_1(data_set_path: str,
           output_bucket_name: str,
           output_file_path_analytical_data_type: str,
           sep: str = ',',
           max_categories: int = 50,
           volume: dsl.VolumeOp = None,
           volume_dir: str = '/mnt'
           ) -> dsl.ContainerOp:
    """
    Generate container operator for task 1

    :param data_set_path: str
        Complete file path of the data set

    :param output_bucket_name: str
        Name of the output S3 bucket

    :param output_file_path_analytical_data_type: str
        Path of the analytical data type information to save

    :param sep: str
        Separator

    :param max_categories: int
        Maximum number of categories for identifying feature as categorical

    :param volume: dsl.VolumeOp
        Attached container volume

    :param volume_dir: str
        Name of the volume directory

    :return: dsl.ContainerOp
        Container operator for task 1
    """
    _volume: dict = {volume_dir: volume if volume is None else volume.volume}
    _task: dsl.ContainerOp = dsl.ContainerOp(name='analytical_data_type',
                                             image='711117404296.dkr.ecr.eu-central-1.amazonaws.com/ml-ops-analytical-data-types:v17',
                                             command=["python", "task.py"],
                                             arguments=['-data_set_path', data_set_path,
                                                        '-output_bucket_name', output_bucket_name,
                                                        '-output_file_path_analytical_data_type', output_file_path_analytical_data_type,
                                                        '-sep', sep,
                                                        '-max_categories', max_categories
                                                        ],
                                             init_containers=None,
                                             sidecars=None,
                                             container_kwargs=None,
                                             artifact_argument_paths=None,
                                             file_outputs={'analytical_data_type': output_file_path_analytical_data_type.split('/')[-1]},
                                             output_artifact_paths=None,
                                             is_exit_handler=False,
                                             pvolumes=volume if volume is None else _volume
                                             )
    _task.set_display_name('Analytical Data Type')
    add_container_op_parameters(container_op=_task,
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


def task_2(data_set_path: str,
           analytical_data_types: dict,
           output_bucket_name: str,
           output_file_path_data_health_check: str,
           output_file_path_data_missing_data: str,
           output_file_path_data_invariant: str,
           output_file_path_data_duplicated: str,
           output_file_path_data_valid_features: str,
           output_file_path_data_prop_valid_features: str,
           sep: str = ',',
           missing_value_threshold: float = 0.8,
           volume: dsl.VolumeOp = None,
           volume_dir: str = '/mnt'
           ) -> dsl.ContainerOp:
    """
    Generate container operator for task 2

    :param data_set_path: str
        Complete file path of the data set

    :param analytical_data_types: dict
        Assign analytical data types to each feature

    :param output_file_path_data_health_check: str
        Path of the data health check results to save

    :param output_file_path_data_missing_data: str
        Path of the features containing too much missing data

    :param output_file_path_data_invariant: str
        Path of the invariant features

    :param output_file_path_data_duplicated: str
        Path of the duplicated features

    :param output_file_path_data_valid_features: str
        Path of the valid features

    :param output_file_path_data_prop_valid_features: str
        Path of the proportion of valid features

    :param output_bucket_name: str
        Name of the output S3 bucket

    :param sep: str
        Separator

    :param missing_value_threshold: float
        Threshold of missing values to exclude numeric feature

    :param volume: dsl.VolumeOp
        Attached container volume

    :param volume_dir: str
        Name of the volume directory

    :return: dsl.ContainerOp
        Container operator for task 2
    """
    _volume: dict = {volume_dir: volume if volume is None else volume.volume}
    _task: dsl.ContainerOp = dsl.ContainerOp(name='data_health_check',
                                             image='711117404296.dkr.ecr.eu-central-1.amazonaws.com/ml-ops-data-health-check:v6',
                                             command=["python", "task.py"],
                                             arguments=['-data_set_path', data_set_path,
                                                        '-analytical_data_types', analytical_data_types,
                                                        '-output_bucket_name', output_bucket_name,
                                                        '-output_file_path_data_health_check', output_file_path_data_health_check,
                                                        '-output_file_path_data_missing_data', output_file_path_data_missing_data,
                                                        '-output_file_path_data_invariant', output_file_path_data_invariant,
                                                        '-output_file_path_data_duplicated', output_file_path_data_duplicated,
                                                        '-output_file_path_data_valid_features', output_file_path_data_valid_features,
                                                        '-output_file_path_data_prop_valid_features', output_file_path_data_prop_valid_features,
                                                        '-sep', sep,
                                                        '-missing_value_threshold', missing_value_threshold
                                                        ],
                                             init_containers=None,
                                             sidecars=None,
                                             container_kwargs=None,
                                             artifact_argument_paths=None,
                                             file_outputs={'data_health_check': output_file_path_data_health_check.split('/')[-1],
                                                           'missing_data': output_file_path_data_missing_data.split('/')[-1],
                                                           'invariant': output_file_path_data_invariant.split('/')[-1],
                                                           'duplicated': output_file_path_data_duplicated.split('/')[-1],
                                                           'valid_features': output_file_path_data_valid_features.split('/')[-1],
                                                           'prop_valid_features': output_file_path_data_prop_valid_features.split('/')[-1]
                                                           },
                                             output_artifact_paths=None,
                                             is_exit_handler=False,
                                             pvolumes=volume if volume is None else _volume
                                             )
    _task.set_display_name('Data Health Check')
    add_container_op_parameters(container_op=_task,
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
    name='PoC',
    description=';-)'
)
def pipeline() -> None:
    """
    Generate Kubeflow Pipeline (v1)
    """
    _data_set_path: str = 's3://shopware-feature-store-prod/external/avocado.csv'
    _output_bucket_name: str = 'shopware-ml-ops-interim-prod'
    _task_1: dsl.ContainerOp = task_1(data_set_path=_data_set_path,
                                      output_bucket_name=_output_bucket_name,
                                      output_file_path_analytical_data_type='external/avocado/analytical_data_type.json',
                                      sep=',',
                                      volume=None,
                                      volume_dir='/mnt'
                                      )
    _task_2: dsl.ContainerOp = task_2(data_set_path=_data_set_path,
                                      analytical_data_types=_task_1.outputs,
                                      output_bucket_name=_output_bucket_name,
                                      output_file_path_data_health_check='external/avocado/data_health_check.json',
                                      missing_value_threshold=0.8,
                                      volume=None,
                                      volume_dir='/mnt'
                                      )
    _task_2.after(_task_1)


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
    _kfp_client.create_experiment(name='avocado_price_prediction',
                                  description='show case: end-to-end ml-ops pipeline for structured data',
                                  namespace=os.getenv('KF_USER_NAMESPACE')
                                  )
    _recurring: bool = False#os.getenv('RECURRING', default=True)
    if _recurring:
        _experiment_id: str = _kfp_client.get_experiment(experiment_id=None,
                                                         experiment_name='avocado_price_prediction',
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
                                         pipeline_package_path='avocado_price_prediction.yaml',
                                         pipeline_id=None,
                                         version_id=None,
                                         enabled=True,
                                         enable_caching=True,
                                         service_account=None
                                         )
    else:
        _kfp_client.create_run_from_pipeline_func(pipeline_func=pipeline,
                                                  arguments={},
                                                  run_name='test2',
                                                  experiment_name='avocado_price_prediction',
                                                  namespace=os.getenv('KF_USER_NAMESPACE'),
                                                  pipeline_root=None,
                                                  enable_caching=False,
                                                  service_account=None
                                                  )
