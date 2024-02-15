"""

Kubeflow Pipeline Component: Machine Learning Model Evaluation

"""

from .container_op_parameters import add_container_op_parameters
from kfp import dsl
from typing import Dict, List

ML_OUTPUT_METRICS: Dict[str, Dict[str, str]] = dict(clf_binary=dict(accuracy='accuracy_metric.json',
                                                                    precision='precision_metric.json',
                                                                    recall='recall_metric.json',
                                                                    f1='f1_metric.json',
                                                                    mcc='mcc_metric.json',
                                                                    roc_auc='roc_auc_metric.json'
                                                                    ),
                                                    clf_multi=dict(cohen_kappa='cohen_kappa_metric.json'),
                                                    reg=dict(r2='r2_metric.json',
                                                             rmse_norm='rmse_norm.json'
                                                             )
                                                    )


def evaluate_machine_learning(ml_type: str,
                              target_feature_name: str,
                              prediction_feature_name: str,
                              train_data_set_path: str,
                              test_data_set_path: str,
                              s3_output_path_metrics: str,
                              s3_output_path_visualization: str = None,
                              s3_output_path_confusion_matrix: str = None,
                              s3_output_path_roc_curve: str = None,
                              s3_output_path_metric_table: str = None,
                              metrics: List[str] = None,
                              labels: List[str] = None,
                              model_id: str = None,
                              val_data_set_path: str = None,
                              sep: str = ',',
                              output_path_metadata: str = 'metadata.json',
                              output_path_sml_score_metric: str = 'sml_score_metric.json',
                              aws_account_id: str = '711117404296',
                              docker_image_name: str = 'ml-ops-model-evaluation',
                              docker_image_tag: str = 'v1',
                              volume: dsl.VolumeOp = None,
                              volume_dir: str = '/mnt',
                              display_name: str = 'Model Evaluation',
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
    Evaluate supervised machine learning models using most common metrics

    :param ml_type: str
        Name of the machine learning problem
            -> reg: Regression
            -> clf_binary: Binary Classification
            -> clf_multi: Multi-Classification
            -> cluster: Clustering

    :param target_feature_name: str
        Name of the target feature

    :param prediction_feature_name: str
        Name of the variable containing predictions

    :param train_data_set_path: str
        Complete file path of the training data set

    :param test_data_set_path: str
        Complete file path of the tests data set

    :param output_path_metadata: str
        File path of the metadata output file

    :param output_path_sml_score_metric: str
        Path of the output sml test metric results to save

    :param s3_output_path_metrics: str
        Complete file path of the metrics output

    :param s3_output_path_visualization: str
        Complete file path of the visualization output

    :param s3_output_path_confusion_matrix: str
        Complete file path of the confusion matrix output

    :param s3_output_path_roc_curve: str
        Complete file path of the roc curve output

    :param s3_output_path_metric_table: str
        Complete file path of the metrics table output

    :param metrics: List[str]
        Abbreviated names of metrics to apply

    :param labels: List[str]
        Class labels of the target feature

    :param model_id: str
        Model identifier

    :param val_data_set_path: str
        Complete file path of the validation data set

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
        Container operator for supervised machine learning model generator
    """
    _volume: dict = {volume_dir: volume if volume is None else volume.volume}
    _arguments: list = ['-ml_type', ml_type,
                        '-target_feature_name', target_feature_name,
                        '-prediction_feature_name', prediction_feature_name,
                        '-train_data_set_path', train_data_set_path,
                        '-test_data_set_path', test_data_set_path,
                        '-s3_output_path_metrics', s3_output_path_metrics,
                        '-sep', sep,
                        '-output_path_metadata', output_path_metadata,
                        '-output_path_r2_metric', ML_OUTPUT_METRICS['reg']['r2'],
                        '-output_path_rmse_norm_metric', ML_OUTPUT_METRICS['reg']['rmse_norm'],
                        '-output_path_accuracy_metric', ML_OUTPUT_METRICS['clf_binary']['accuracy'],
                        '-output_path_precision_metric', ML_OUTPUT_METRICS['clf_binary']['precision'],
                        '-output_path_recall_metric', ML_OUTPUT_METRICS['clf_binary']['recall'],
                        '-output_path_f1_metric', ML_OUTPUT_METRICS['clf_binary']['f1'],
                        '-output_path_mcc_metric', ML_OUTPUT_METRICS['clf_binary']['mcc'],
                        '-output_path_roc_auc_metric', ML_OUTPUT_METRICS['clf_binary']['roc_auc'],
                        '-output_path_cohen_kappa_metric', ML_OUTPUT_METRICS['clf_multi']['cohen_kappa'],
                        '-output_path_sml_score_metric', output_path_sml_score_metric,
                        ]
    if s3_output_path_visualization is not None:
        _arguments.extend(['-s3_output_path_visualization', s3_output_path_visualization])
    if s3_output_path_confusion_matrix is not None:
        _arguments.extend(['-s3_output_path_confusion_matrix', s3_output_path_confusion_matrix])
    if s3_output_path_roc_curve is not None:
        _arguments.extend(['-s3_output_path_roc_curve', s3_output_path_roc_curve])
    if s3_output_path_metric_table is not None:
        _arguments.extend(['-s3_output_path_metric_table', s3_output_path_metric_table])
    if val_data_set_path is not None:
        _arguments.extend(['-val_data_set_path', val_data_set_path])
    if metrics is not None:
        _arguments.extend(['-metrics', metrics])
    if labels is not None:
        _arguments.extend(['-labels', labels])
    if model_id is not None:
        _arguments.extend(['-model_id', model_id])
    _file_outputs: Dict[str, str] = dict(metadata=output_path_metadata,
                                         sml_score=output_path_sml_score_metric
                                         )
    for output_metric in ML_OUTPUT_METRICS[ml_type].keys():
        if metrics is not None:
            if output_metric not in metrics:
                continue
        _file_outputs.update({output_metric: ML_OUTPUT_METRICS[ml_type][output_metric]})
    _task: dsl.ContainerOp = dsl.ContainerOp(name='model_evaluation',
                                             image=f'{aws_account_id}.dkr.ecr.eu-central-1.amazonaws.com/{docker_image_name}:{docker_image_tag}',
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
