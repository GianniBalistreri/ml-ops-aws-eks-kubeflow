"""

Task: ... (Function to run in container)

"""

import argparse
import ast
import base64
import pandas as pd

from aws import load_file_from_s3, load_file_from_s3_as_df, save_file_to_s3
from custom_logger import Log
from file_handler import file_handler
from interactive_visualizer import InteractiveVisualizer, InteractiveVisualizerException
from resource_metrics import get_available_cpu, get_cpu_utilization, get_cpu_utilization_per_core, get_memory, get_memory_utilization
from typing import Dict, List, NamedTuple

PARSER = argparse.ArgumentParser(description="interactive visualizer")
PARSER.add_argument('-data_set_path', type=str, required=False, default=None, help='file path of the data set')
PARSER.add_argument('-plot_type', type=str, required=False, default=None, help='abbreviated name of the plot type to use')
PARSER.add_argument('-interactive', type=int, required=False, default=1, help='whether to generate interactive plots or static')
PARSER.add_argument('-title', type=str, required=False, default='', help='title of the visualization')
PARSER.add_argument('-features', nargs='+', required=False, default=None, help='feature names to visualize')
PARSER.add_argument('-time_features', nargs='+', required=False, default=None, help='feature names used as grouping features by time')
PARSER.add_argument('-graph_features', nargs='+', required=False, default=None, help='feature names used to build network graph')
PARSER.add_argument('-group_by', nargs='+', required=False, default=None, help='feature names used as grouping feature')
PARSER.add_argument('-melt', type=int, required=False, default=0, help='whether to combine group by plots')
PARSER.add_argument('-brushing', type=int, required=False, default=0, help='whether to use brushing for parallel category plot or not')
PARSER.add_argument('-xaxis_label', nargs='+', required=False, default=None, help='Labels for the x-axis')
PARSER.add_argument('-yaxis_label', nargs='+', required=False, default=None, help='Labels for the y-axis')
PARSER.add_argument('-zaxis_label', nargs='+', required=False, default=None, help='Labels for the z-axis')
PARSER.add_argument('-annotations', nargs='+', required=False, default=None, help='annotations used to enrich the visualization with additional information')
PARSER.add_argument('-width', type=int, required=False, default=500, help='Width of the plot')
PARSER.add_argument('-height', type=int, required=False, default=500, help='Height of the plot')
PARSER.add_argument('-unit', type=str, required=False, default='px', help='measurement unit for the plot size')
PARSER.add_argument('-use_auto_extensions', type=int, required=False, default=0, help='whether to use automatic file name extensions when generate grouping visualizations or not')
PARSER.add_argument('-color_scale', nargs='+', required=False, default=None, help='color names')
PARSER.add_argument('-color_edges', nargs='+', required=False, default=None, help='color names of the color edges')
PARSER.add_argument('-color_feature', type=str, required=False, default=None, help='feature name to display values in specific color range')
PARSER.add_argument('-analytical_data_types_path', type=str, required=False, default=None, help='assignment of features to analytical data types')
PARSER.add_argument('-subplots_file_path', type=str, required=False, default=None, help='complete file path of the subplots configuration')
PARSER.add_argument('-output_file_paths', type=str, required=True, default=None, help='file path of the visualization file paths output')
PARSER.add_argument('-sep', type=str, required=False, default=',', help='column separator')
PARSER.add_argument('-s3_output_image_path', type=str, required=True, default=None, help='complete file path of the visualization output')
ARGS = PARSER.parse_args()


def _convert_to_html(file_path: str) -> str:
    """
    Convert static plot (png, jpg, jpeg) to html file

    :param file_path: str
        Complete file path of the static plot

    :return str
        Complete file path of the converted image
    """
    with open(file_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        _base64_image: str = f"data:image/png;base64,{encoded_image}"
    _html_content: str = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body>
    <img src="{_base64_image}">
    </body>
    </html>
    """
    _file_type: str = file_path.split('.')[-1]
    _file_path: str = file_path.replace(_file_type, 'html')
    save_file_to_s3(file_path=_file_path, obj=_html_content, plotly=False)
    return _file_path


def interactive_visualizer(s3_output_image_path: str,
                           output_file_paths: str,
                           data_set_path: str = None,
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
                           analytical_data_types_path: str = None,
                           subplots_file_path: str = None,
                           sep: str = ','
                           ) -> NamedTuple('outputs', [('plot_file_paths', list)]):
    """
    Interactive visualization

    :param s3_output_image_path: str
        Complete file path of the image output

    :param output_file_paths: str
        File path of the visualization output file paths

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
                -> line: Line plot
                -> pie: Pie plot
                -> scatter: Scatter plot
                -> table: Table plot

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

    :return: NamedTuple
        Paths of the visualization
    """
    _cpu_available: int = get_available_cpu(logging=True)
    _memory_total: float = get_memory(total=True, logging=True)
    _memory_available: float = get_memory(total=False, logging=True)
    if data_set_path is None:
        _df: pd.DataFrame = None
        if subplots_file_path is None:
            raise InteractiveVisualizerException('Neither path of the data set nor path of the subplots found')
        else:
            _subplots: dict = load_file_from_s3(file_path=subplots_file_path)
            Log().log(msg=f'Load ({len(_subplots.keys())}) subplot configurations: {subplots_file_path}')
    else:
        _subplots: dict = None
        _df: pd.DataFrame = load_file_from_s3_as_df(file_path=data_set_path, sep=sep)
        Log().log(msg=f'Load data set: {data_set_path} -> Cases={_df.shape[0]}, Features={_df.shape[1]}')
    if analytical_data_types_path is None:
        _analytical_data_types: Dict[str, List[str]] = None
    else:
        _analytical_data_types: Dict[str, List[str]] = load_file_from_s3(file_path=analytical_data_types_path)
        Log().log(msg=f'Load analytical data types: {analytical_data_types_path}')
    _interactive_visualizer: InteractiveVisualizer = InteractiveVisualizer(df=_df,
                                                                           title=title,
                                                                           features=features,
                                                                           time_features=time_features,
                                                                           graph_features=graph_features,
                                                                           group_by=group_by,
                                                                           feature_types=_analytical_data_types,
                                                                           plot_type=plot_type,
                                                                           subplots=_subplots,
                                                                           melt=melt,
                                                                           brushing=brushing,
                                                                           xaxis_label=xaxis_label,
                                                                           yaxis_label=yaxis_label,
                                                                           zaxis_label=zaxis_label,
                                                                           annotations=annotations,
                                                                           width=width,
                                                                           height=height,
                                                                           unit=unit,
                                                                           file_path=s3_output_image_path,
                                                                           use_auto_extensions=use_auto_extensions,
                                                                           cloud='aws',
                                                                           render=False,
                                                                           color_scale=color_scale,
                                                                           color_edges=color_edges,
                                                                           color_feature=color_feature,
                                                                           max_row=50,
                                                                           max_col=20,
                                                                           rows_sub=None,
                                                                           cols_sub=None
                                                                           )
    _file_paths: List[str] = _interactive_visualizer.main()
    _new_file_paths: List[str] = []
    for file_path in _file_paths:
        _file_type: str = file_path.split('.')[-1]
        if _file_type != 'html' and not interactive:
            _new_file_paths.append(_convert_to_html(file_path=file_path))
        else:
            _new_file_paths.append(file_path)
    file_handler(file_path=output_file_paths, obj=_new_file_paths)
    _cpu_utilization: float = get_cpu_utilization(interval=1, logging=True)
    _cpu_utilization_per_cpu: List[float] = get_cpu_utilization_per_core(interval=1, logging=True)
    _memory_utilization: float = get_memory_utilization(logging=True)
    _memory_available = get_memory(total=False, logging=True)
    return [_new_file_paths]


if __name__ == '__main__':
    if ARGS.features:
        ARGS.features = ast.literal_eval(ARGS.features[0])
    if ARGS.time_features:
        ARGS.time_features = ast.literal_eval(ARGS.time_features[0])
    if ARGS.graph_features:
        ARGS.graph_features = ast.literal_eval(ARGS.graph_features[0])
    if ARGS.group_by:
        ARGS.group_by = ast.literal_eval(ARGS.group_by[0])
    if ARGS.xaxis_label:
        ARGS.xaxis_label = ast.literal_eval(ARGS.xaxis_label[0])
    if ARGS.yaxis_label:
        ARGS.yaxis_label = ast.literal_eval(ARGS.yaxis_label[0])
    if ARGS.zaxis_label:
        ARGS.zaxis_label = ast.literal_eval(ARGS.zaxis_label[0])
    if ARGS.annotations:
        ARGS.annotations = ast.literal_eval(ARGS.annotations[0])
    interactive_visualizer(s3_output_image_path=ARGS.s3_output_image_path,
                           output_file_paths=ARGS.output_file_paths,
                           data_set_path=ARGS.data_set_path,
                           plot_type=ARGS.plot_type,
                           interactive=bool(ARGS.interactive),
                           title=ARGS.title,
                           features=ARGS.features,
                           time_features=ARGS.time_features,
                           graph_features=ARGS.graph_features,
                           group_by=ARGS.group_by,
                           analytical_data_types_path=ARGS.analytical_data_types_path,
                           melt=bool(ARGS.melt),
                           brushing=bool(ARGS.brushing),
                           xaxis_label=ARGS.xaxis_label,
                           yaxis_label=ARGS.yaxis_label,
                           zaxis_label=ARGS.zaxis_label,
                           annotations=ARGS.annotations,
                           width=ARGS.width,
                           height=ARGS.height,
                           unit=ARGS.unit,
                           use_auto_extensions=bool(ARGS.use_auto_extensions),
                           color_scale=ARGS.color_scale,
                           color_edges=ARGS.color_edges,
                           color_feature=ARGS.color_feature,
                           subplots_file_path=ARGS.subplots_file_path,
                           sep=ARGS.sep,
                           )
