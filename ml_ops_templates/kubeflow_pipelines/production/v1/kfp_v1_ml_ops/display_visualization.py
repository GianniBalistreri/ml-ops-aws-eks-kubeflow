"""

Kubeflow Pipeline Component: Display Visualization

"""

import kfp

from .container_op_parameters import add_container_op_parameters
from kfp import dsl
from kfp.components import create_component_from_func
from typing import Dict, List, Union


class DisplayVisualizationException(Exception):
    """
    Class for handling exceptions for function display_visualization
    """
    pass


def _generate_kfp_web_app_template(mlpipeline_ui_metadata_path: kfp.components.OutputPath(), file_paths: str):
    """
    Generate Kubeflow pipeline visualization template

    :param mlpipeline_ui_metadata_path: kfp.components.OutputPath()
        Kubeflow pipelines OutputPath component

    :param file_paths: str
        Complete file path of the plots
    """
    import ast
    import json
    from typing import Dict, List
    _file_paths: Dict[str, List[str]] = ast.literal_eval(file_paths)
    _metadata: Dict[str, List[Dict[str, str]]] = dict(outputs=[])
    for key in _file_paths.keys():
        for file_path in _file_paths[key]:
            _plot_config: Dict[str, str] = dict(type='web-app', storage='s3', source=file_path)
            _metadata['outputs'].append(_plot_config)
    with open(mlpipeline_ui_metadata_path, 'w') as _file:
        json.dump(_metadata, _file)


def display_visualization(file_paths: Union[Dict[str, List[str]], dsl.PipelineParam],
                          python_version: str = '3.9',
                          display_name: str = 'Display Visualization',
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
    Display visualization

    :param file_paths: Union[Dict[str, List[str]], dsl.PipelineParam]
        Complete file path of the plots

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
    _container_from_func: dsl.component = create_component_from_func(func=_generate_kfp_web_app_template,
                                                                     output_component_file=None,
                                                                     base_image=f'python:{python_version}',
                                                                     packages_to_install=None,
                                                                     annotations=None
                                                                     )
    _task: dsl.ContainerOp = _container_from_func(file_paths=str(file_paths))
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
