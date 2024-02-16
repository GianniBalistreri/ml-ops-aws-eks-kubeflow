"""

Kubeflow Pipeline Component: Supervised Machine Learning Model Generator (structured data)

"""

from .container_op_parameters import add_container_op_parameters
from kfp import dsl
from typing import Any, List, Union


def generate_supervised_model(ml_type: Union[str, dsl.PipelineParam],
                              model_name: Union[str, dsl.PipelineParam],
                              target_feature: Union[str, dsl.PipelineParam],
                              train_data_set_path: Union[str, dsl.PipelineParam],
                              test_data_set_path: Union[str, dsl.PipelineParam],
                              s3_output_path_model: Union[str, dsl.PipelineParam],
                              s3_output_path_param: Union[str, dsl.PipelineParam],
                              s3_output_path_metadata: Union[str, dsl.PipelineParam],
                              s3_output_path_evaluation_train_data: Union[str, dsl.PipelineParam],
                              s3_output_path_evaluation_test_data: Union[str, dsl.PipelineParam],
                              aws_account_id: str,
                              aws_region: str,
                              predictors: Union[List[str], dsl.PipelineParam] = None,
                              model_id: Union[int, dsl.PipelineParam] = None,
                              model_param_path: Union[str, dsl.PipelineParam] = None,
                              param_rate: Union[float, dsl.PipelineParam] = 0.0,
                              force_param_path: str = None,
                              warm_start: Any = 1,
                              max_retries: int = 100,
                              train_model: bool = True,
                              sep: str = ',',
                              prediction_variable_name: str = 'prediction',
                              parallel_mode: bool = False,
                              val_data_set_path: Union[str, dsl.PipelineParam] = None,
                              s3_output_path_evaluation_val_data: Union[str, dsl.PipelineParam] = None,
                              output_path_training_status: str = 'training_status.json',
                              docker_image_name: str = 'ml-ops-model-generator-supervised',
                              docker_image_tag: str = 'v1',
                              volume: dsl.VolumeOp = None,
                              volume_dir: str = '/mnt',
                              display_name: str = 'Supervised Model Generator',
                              n_cpu_request: str = None,
                              n_cpu_limit: str = None,
                              n_gpu: str = None,
                              gpu_vendor: str = 'nvidia',
                              memory_request: str = '1G',
                              memory_limit: str = None,
                              ephemeral_storage_request: str = '5G',
                              ephemeral_storage_limit: str = None,
                              instance_name: str = 'm5.xlarge',
                              max_cache_staleness: str = 'P0D',
                              **kwargs
                              ) -> dsl.ContainerOp:
    """
    Generate supervised machine learning model for structured data

    :param ml_type: str
        Name of the machine learning problem
            -> reg: Regression
            -> clf_binary: Binary Classification
            -> clf_multi: Multi-Classification

    :param model_name: str
        Abbreviated name of the supervised machine learning model

    :param target_feature: str
        Name of the target feature

    :param train_data_set_path: str
        Complete file path of the training data set

    :param test_data_set_path: str
        Complete file path of the tests data set

    :param s3_output_path_model: str
        Complete S3 file path of the trained model artifact

    :param s3_output_path_param: str
        Complete file path of the hyperparameter output

    :param s3_output_path_metadata: str
        Complete file path of the metadata output

    :param s3_output_path_evaluation_train_data: str
        Complete file path of the evaluation training data set output

    :param s3_output_path_evaluation_test_data: str
        Complete file path of the evaluation test data set output

    :param aws_account_id: str
        AWS account id

    :param aws_region: str
        AWS region name

    :param predictors: List[str]
        Name of the predictors

    :param model_id: int
        Model ID

    :param model_param_path: str
        Complete file path of the model hyperparameter set

    :param param_rate: float
        Rate for changing given hyperparameter set

    :param force_param_path: str
        Complete file path of the immutable model hyperparameter set

    :param warm_start: Any
        Whether to use standard hyperparameter set or not

    :param max_retries: int
        Maximum number of retries if model hyperparameter configuration raises an error

    :param train_model: bool
        Whether to train model or not

    :param sep: str
        Separator

    :param prediction_variable_name: str
        Name of the prediction variable for evaluation step afterward

    :param parallel_mode: bool
        Whether to run task in parallel mode or not

    :param val_data_set_path: str
        Complete file path of the validation data set

    :param s3_output_path_evaluation_val_data: str
        Complete file path of the evaluation validation data set output

    :param output_path_training_status: str
        Path of the training status output

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

    :param kwargs: dict
        Key-word arguments for handling low and high boundaries for randomly drawing model hyperparameter configuration

    :return: dsl.ContainerOp
        Container operator for supervised machine learning model generator
    """
    _volume: dict = {volume_dir: volume if volume is None else volume.volume}
    _arguments: list = ['-ml_type', ml_type,
                        '-model_name', model_name,
                        '-target_feature', target_feature,
                        '-train_data_set_path', train_data_set_path,
                        '-test_data_set_path', test_data_set_path,
                        '-s3_output_path_model', s3_output_path_model,
                        '-s3_output_path_metadata', s3_output_path_metadata,
                        '-s3_output_path_evaluation_train_data', s3_output_path_evaluation_train_data,
                        '-s3_output_path_evaluation_test_data', s3_output_path_evaluation_test_data,
                        '-output_path_training_status', output_path_training_status,
                        '-param_rate', param_rate,
                        '-warm_start', warm_start,
                        '-max_retries', max_retries,
                        '-train_model', int(train_model),
                        '-sep', sep,
                        '-prediction_variable_name', prediction_variable_name,
                        '-parallel_mode', int(parallel_mode)
                        ]
    if predictors is not None:
        _arguments.extend(['-predictors', predictors])
    if model_id is not None:
        _arguments.extend(['-model_id', model_id])
    if model_param_path is not None:
        _arguments.extend(['-model_param_path', model_param_path])
    if force_param_path is not None:
        _arguments.extend(['-force_param_path', force_param_path])
    if val_data_set_path is not None:
        _arguments.extend(['-val_data_set_path', val_data_set_path])
    if s3_output_path_param is not None:
        _arguments.extend(['-s3_output_path_param', s3_output_path_param])
    if s3_output_path_evaluation_val_data is not None:
        _arguments.extend(['-s3_output_path_evaluation_val_data', s3_output_path_evaluation_val_data])
    if kwargs is not None:
        _arguments.extend(['-kwargs', kwargs])
    _task: dsl.ContainerOp = dsl.ContainerOp(name='supervised_model_generator',
                                             image=f'{aws_account_id}.dkr.ecr.{aws_region}.amazonaws.com/{docker_image_name}:{docker_image_tag}',
                                             command=["python", "task.py"],
                                             arguments=_arguments,
                                             init_containers=None,
                                             sidecars=None,
                                             container_kwargs=None,
                                             artifact_argument_paths=None,
                                             file_outputs={'training_status': output_path_training_status},
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
