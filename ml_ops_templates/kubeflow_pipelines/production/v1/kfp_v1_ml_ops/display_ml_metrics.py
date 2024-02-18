"""

Kubeflow Pipeline Component: Display Kubeflow Pipeline Metrics

"""

import kfp

from .container_op_parameters import add_container_op_parameters
from kfp import dsl
from kfp.components import create_component_from_func
from typing import List, Union

KFP_BUILD_IN_METRIC_TYPE: List[str] = ['accuracy-score', 'roc-auc-score']
KFP_BUILD_IN_METRIC_VISUALIZATION_TYPE: List[str] = ['confusion_matrix', 'roc', 'table']


class DisplayMetricsException(Exception):
    """
    Class for handling exceptions for function display_visualization
    """
    pass


def _generate_kfp_metric_template(mlpipeline_ui_metadata_path: kfp.components.OutputPath(),
                                  file_paths: str,
                                  metric_types: str,
                                  metric_values: str = None,
                                  metric_formats: str = None,
                                  target_feature: str = None,
                                  prediction_feature: str = None,
                                  labels: str = None,
                                  header: str = None
                                  ):
    """
    Generate Kubeflow pipeline visualization template

    :param file_paths: List[str]
        Complete file path of the plots

    :param metric_types: List[str]
        Name of the pre-defined kfp visualization
            -> confusion_matrix: Confusion matrix
            -> roc: ROC-AUC
            -> table: Data table
            -> accuracy-score: Accuracy
            -> roc-auc-score: ROC-AUC

    :param metric_values: List[float]
        Metric value

    :param metric_formats: List[str]
        Value formats of the Kubeflow pipeline metric
            -> RAW: raw value
            -> PERCENTAGE: scaled percentage value

    :param target_feature: str
        Name of the target feature

    :param prediction_feature: str
        Name of the prediction feature

    :param labels: List[str]
        Labels of the target feature

    :param header: List[str]
        Table headers

    :return: Dict[str, List[Dict[str, str]]]
        Configured Kubeflow pipeline metadata template
    """
    import ast
    import json
    from datetime import datetime
    from typing import Dict, List
    _logger_time: str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    _kfp_build_in_metric_type: List[str] = ['accuracy-score', 'roc-auc-score']
    _i: int = 0
    _file_paths: List[str] = ast.literal_eval(file_paths)
    _metric_types: List[str] = ast.literal_eval(metric_types)
    _metric_values: List[float] = [] if metric_values is None else ast.literal_eval(metric_values)
    _metric_formats: List[str] = [] if metric_formats is None else ast.literal_eval(metric_formats)
    _labels: List[str] = [] if labels is None else ast.literal_eval(labels)
    _header: List[str] = [] if header is None else ast.literal_eval(header)
    _metadata: Dict[str, List[Dict[str, str]]] = {}
    for file_path, metric_type in zip(_file_paths, _metric_types):
        if metric_type in _kfp_build_in_metric_type:
            if 'metrics' not in _metadata.keys():
                _metadata.update(dict(metrics=[]))
            _metric_type: str = 'metrics'
        else:
            if 'outputs' not in _metadata.keys():
                _metadata.update(dict(outputs=[]))
            _metric_type: str = 'outputs'
        if metric_type == 'confusion_matrix':
            _schema: List[Dict[str, str]] = list()
            _schema.append(dict(name=target_feature, type='CATEGORY'))
            _schema.append(dict(name=prediction_feature, type='CATEGORY'))
            _schema.append(dict(name='count', type='NUMBER'))
            _plot_config: Dict[str, str] = dict(type=metric_type,
                                                format='csv',
                                                schema=_schema,
                                                labels=_labels,
                                                storage='s3',
                                                source=file_path
                                                )
        elif metric_type == 'roc':
            _schema: List[Dict[str, str]] = list()
            _schema.append(dict(name='fpr', type='NUMBER'))
            _schema.append(dict(name='tpr', type='NUMBER'))
            _schema.append(dict(name='thresholds', type='NUMBER'))
            _plot_config: Dict[str, str] = dict(type=metric_type,
                                                format='csv',
                                                schema=_schema,
                                                target_col=target_feature,
                                                predicted_col=prediction_feature,
                                                storage='s3',
                                                source=file_path
                                                )
        elif metric_type == 'table':
            _plot_config: Dict[str, str] = dict(type=metric_type,
                                                format='csv',
                                                header=_header,
                                                storage='s3',
                                                source=file_path
                                                )
        else:
            _plot_config: Dict[str, str] = dict(name=metric_type, format=_metric_formats[_i], numberValue=metric_values)
        _metadata[_metric_type].append(_plot_config)
        _i += 1
    print(f'{_logger_time} Kubeflow metric config: {_metadata}')
    with open(mlpipeline_ui_metadata_path, 'w') as _file:
        json.dump(_metadata, _file)


def display_metrics(file_paths: Union[List[str], dsl.PipelineParam],
                    metric_types: List[str],
                    metric_values: List[float] = None,
                    metric_formats: List[str] = None,
                    target_feature: str = None,
                    prediction_feature: str = None,
                    labels: List[str] = None,
                    header: List[str] = None,
                    python_version: str = '3.9',
                    display_name: str = 'Display Metrics',
                    n_cpu_request: str = None,
                    n_cpu_limit: str = None,
                    n_gpu: str = None,
                    gpu_vendor: str = 'nvidia',
                    memory_request: str = '100Mi',
                    memory_limit: str = None,
                    ephemeral_storage_request: str = '100Mi',
                    ephemeral_storage_limit: str = None,
                    instance_name: str = 'm5.xlarge',
                    max_cache_staleness: str = 'P0D'
                    ) -> dsl.ContainerOp:
    """
    Display metrics

    :param file_paths: Union[List[str], dsl.PipelineParam]
        Complete file path of the plots

    :param metric_types: List[str]
        Name of the pre-defined kfp visualization
            -> confusion_matrix: Confusion matrix
            -> roc: ROC-AUC
            -> table: Data table

    :param metric_values: List[float]
        Metric value

    :param metric_formats: List[str]
        Value formats of the Kubeflow pipeline metric
            -> RAW: raw value
            -> PERCENTAGE: scaled percentage value

    :param target_feature: str
        Name of the target feature

    :param prediction_feature: str
        Name of the prediction feature

    :param labels: List[str]
        Labels of the target feature

    :param header: List[str]
        Table headers

    :param python_version: str
        Python version of the base image

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
    for metric_type in metric_types:
        if metric_type not in KFP_BUILD_IN_METRIC_TYPE:
            if metric_type not in KFP_BUILD_IN_METRIC_VISUALIZATION_TYPE:
                raise DisplayMetricsException(f'Kubeflow pipeline visualization ({metric_type}) not supported')
    _container_from_func: dsl.component = create_component_from_func(func=_generate_kfp_metric_template,
                                                                     output_component_file=None,
                                                                     base_image=f'python:{python_version}',
                                                                     packages_to_install=None,
                                                                     annotations=None
                                                                     )
    _task: dsl.ContainerOp = _container_from_func(file_paths=str(file_paths),
                                                  metric_types=str(metric_types),
                                                  metric_values=None if metric_values is None else str(metric_values),
                                                  metric_formats=None if metric_formats is None else str(metric_formats),
                                                  target_feature=target_feature,
                                                  prediction_feature=prediction_feature,
                                                  labels=None if labels is None else str(labels),
                                                  header=None if header is None else str(header)
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
