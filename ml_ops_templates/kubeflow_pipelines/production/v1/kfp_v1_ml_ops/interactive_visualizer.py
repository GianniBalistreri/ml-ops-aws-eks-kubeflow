"""

Kubeflow Pipeline Component: Interactive Visualizer

"""

from .container_op_parameters import add_container_op_parameters
from kfp import dsl
from typing import List, Union


def interactive_visualizer(s3_output_image_path: Union[str, dsl.PipelineParam],
                           aws_account_id: str,
                           aws_region: str,
                           data_set_path: Union[str, dsl.PipelineParam] = None,
                           plot_type: str = None,
                           interactive: bool = True,
                           title: str = None,
                           features: List[str] = None,
                           time_features: List[str] = None,
                           graph_features: List[str] = None,
                           group_by: List[str] = None,
                           melt: bool = False,
                           brushing: bool = False,
                           xaxis_label: List[str] = None,
                           yaxis_label: List[str] = None,
                           zaxis_label: List[str] = None,
                           annotations: List[dict] = None,
                           width: int = 500,
                           height: int = 500,
                           unit: str = 'px',
                           use_auto_extensions: bool = False,
                           color_scale: List[str] = None,
                           color_edges: List[str] = None,
                           color_feature: str = None,
                           analytical_data_types_path: Union[str, dsl.PipelineParam] = None,
                           subplots_file_path: Union[str, dsl.PipelineParam] = None,
                           sep: str = ',',
                           output_file_paths: str = 'file_paths.json',
                           docker_image_name: str = 'ml-ops-interactive-visualizer',
                           docker_image_tag: str = 'v1',
                           volume: dsl.VolumeOp = None,
                           volume_dir: str = '/mnt',
                           display_name: str = 'Interactive Visualizer',
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
    Interactive visualization

    :param s3_output_image_path: str
        Name of the output destination of the image

    :param aws_account_id: str
        AWS account id

    :param aws_region: str
        AWS region name

    :param data_set_path: str
        Complete file path of the data set

    :param plot_type: str
            Name of the plot type
                -> bar: Bar plot
                -> bi: Bivariate plot (categorical - continuous)
                -> box: Box-Whisker plot
                -> geo: Geo map
                -> heat: Heat map
                -> hist: Histogram
                -> joint: Joint plot
                -> line: Line plot
                -> pie: Pie plot
                -> scatter: Scatter plot
                -> table: Table plot
                -> violin: Violin plot

    :param output_file_paths: str
        File path of the visualization output file paths

    :param interactive: bool
        Whether to generate interactive visualization or static

    :param title: str
        Name of the plot title

    :param features: List[str]
        Name of the features

    :param time_features: List[str]
        Name of the time regarding features in line plots

    :param graph_features: List[str]
        Name of the graph features in graph plots

    :param group_by: List[str]
        Name of the categorical features to generate group by plots

    :param melt: bool
        Melt subplots into one main plot

    :param brushing: bool
        Generate additional scatter chart for case-based exploration of feature connections

    :param xaxis_label: List[str]
        User based labeling of the x-axis for all subplots

    :param yaxis_label: List[str]
        User based labeling of the y-axis for all subplots

    :param zaxis_label: List[str]
        User based labeling of the z-axis for all subplots

    :param annotations: List[dict]
        Annotation configuration for each subplot

    :param width: int
        Width size for each subplot

    :param height: int
        Height size for each subplot

    :param unit: str
        Measurement unit
            -> px, pixel: Pixel
            -> in, inch: Inch
            -> cm, centimeter: Centimeter

    :param use_auto_extensions: bool
        Use automatic file name extensions beyond group by functionality

    :param color_scale: List[str]
        Name of the color scale

    :param color_edges: List[str]
        Name of the color edges

    :param color_feature: str
        Name of the feature to display values within specific color range

    :param analytical_data_types_path: str
        Complete file path of the analytical data types

    :param subplots_file_path: str
        Complete file path of the pre-defined subplot configuration

    :param sep: str
        Separator

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
    _arguments: list = ['-output_file_paths', output_file_paths,
                        '-s3_output_image_path', s3_output_image_path,
                        '-interactive', int(interactive),
                        '-melt', int(melt),
                        '-brushing', int(brushing),
                        '-width', width,
                        '-height', height,
                        '-unit', unit,
                        '-use_auto_extensions', int(use_auto_extensions),
                        '-sep', sep
                        ]
    if data_set_path is not None:
        _arguments.extend(['-data_set_path', data_set_path])
    if plot_type is not None:
        _arguments.extend(['-plot_type', plot_type])
    if title is not None:
        _arguments.extend(['-title', title])
    if features is not None:
        _arguments.extend(['-features', features])
    if time_features is not None:
        _arguments.extend(['-time_features', time_features])
    if graph_features is not None:
        _arguments.extend(['-graph_features', graph_features])
    if group_by is not None:
        _arguments.extend(['-group_by', group_by])
    if xaxis_label is not None:
        _arguments.extend(['-xaxis_label', xaxis_label])
    if yaxis_label is not None:
        _arguments.extend(['-yaxis_label', yaxis_label])
    if zaxis_label is not None:
        _arguments.extend(['-zaxis_label', zaxis_label])
    if annotations is not None:
        _arguments.extend(['-annotations', annotations])
    if color_scale is not None:
        _arguments.extend(['-color_scale', color_scale])
    if color_edges is not None:
        _arguments.extend(['-color_edges', color_edges])
    if color_feature is not None:
        _arguments.extend(['-color_feature', color_feature])
    if analytical_data_types_path is not None:
        _arguments.extend(['-analytical_data_types_path', analytical_data_types_path])
    if subplots_file_path is not None:
        _arguments.extend(['-subplots_file_path', subplots_file_path])
    _task: dsl.ContainerOp = dsl.ContainerOp(name='interactive_visualizer',
                                             image=f'{aws_account_id}.dkr.ecr.{aws_region}.amazonaws.com/{docker_image_name}:{docker_image_tag}',
                                             command=["python", "task.py"],
                                             arguments=_arguments,
                                             init_containers=None,
                                             sidecars=None,
                                             container_kwargs=None,
                                             artifact_argument_paths=None,
                                             file_outputs={'file_paths': output_file_paths},
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
