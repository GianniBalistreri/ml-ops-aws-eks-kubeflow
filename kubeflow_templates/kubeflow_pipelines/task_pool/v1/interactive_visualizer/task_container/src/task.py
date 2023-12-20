"""

Task: ... (Function to run in container)

"""

import argparse
import json
import pandas as pd

from interactive_visualizer import InteractiveVisualizer
from kfp.components import OutputPath
from typing import Any, Dict, NamedTuple, List

PARSER = argparse.ArgumentParser(description="interactive visualizer")
PARSER.add_argument('-data_set_path', type=str, required=True, default=None, help='file path of the data set')
PARSER.add_argument('-plot_type', type=str, required=True, default=None, help='abbreviated name of the plot type to use')
PARSER.add_argument('-title', type=str, required=False, default='', help='title of the visualization')
PARSER.add_argument('-feature_names', type=str, required=False, default=None, help='feature names to visualize')
PARSER.add_argument('-time_features', type=str, required=False, default=None, help='feature names used as grouping features by time')
PARSER.add_argument('-graph_features', type=str, required=False, default=None, help='feature names used to build network graph')
PARSER.add_argument('-group_by', type=str, required=False, default=None, help='feature names used as grouping feature')
PARSER.add_argument('-analytical_data_types', type=Any, required=True, default=None, help='assignment of features to analytical data types')
PARSER.add_argument('-melt', type=int, required=False, default=0, help='whether to combine group by plots')
PARSER.add_argument('-brushing', type=int, required=False, default=0, help='whether to use brushing for parallel category plot or not')
PARSER.add_argument('-xaxis_label', type=list, required=False, default=None, help='Labels for the x-axis')
PARSER.add_argument('-yaxis_label', type=list, required=False, default=None, help='Labels for the y-axis')
PARSER.add_argument('-zaxis_label', type=list, required=False, default=None, help='Labels for the z-axis')
PARSER.add_argument('-annotations', type=list, required=True, default=None, help='annotations used to enrich the visualization with additional information')
PARSER.add_argument('-width', type=int, required=False, default=500, help='Width of the plot')
PARSER.add_argument('-height', type=list, required=False, default=500, help='Height of the plot')
PARSER.add_argument('-unit', type=str, required=False, default='px', help='measurement unit for the plot size')
PARSER.add_argument('-use_auto_extensions', type=int, required=False, default=0, help='whether to use automatic file name extensions when generate grouping visualizations or not')
PARSER.add_argument('-zaxis_label', type=list, required=False, default=None, help='Labels for the z-axis')
PARSER.add_argument('-zaxis_label', type=list, required=False, default=None, help='Labels for the z-axis')
PARSER.add_argument('-zaxis_label', type=list, required=False, default=None, help='Labels for the z-axis')
PARSER.add_argument('-zaxis_label', type=list, required=False, default=None, help='Labels for the z-axis')
PARSER.add_argument('-output_image_path', type=str, required=True, default=None, help='complete file path of the visualization output')
PARSER.add_argument('-sep', type=str, required=False, default=',', help='column separator')
ARGS = PARSER.parse_args()


def _generate_kfp_template(file_paths: List[str]) -> Dict[str, List[Dict[str, str]]]:
    """
    Generate Kubeflow pipeline visualization template

    :param file_paths: List[str]
        Complete file path of the plots

    :return: Dict[str, List[Dict[str, str]]]
        Configured Kubeflow pipeline metadata template
    """
    _metadata: Dict[str, List[Dict[str, str]]] = dict(outputs=[])
    for file_path in file_paths:
        _plot_config: Dict[str, str] = dict(type='web-app', storage='s3', source=file_path)
        _metadata['outputs'].append(_plot_config)
    #metadata = {
    #    'outputs' : [{
    #        'type': 'web-app',
    #        'storage': 's3',
    #        'source': 's3://shopware-ml-ops-interim-prod/viz.png',
    #    }, {
    #        'type': 'web-app',
    #        'storage': 'inline',
    #        'source': '<h1>Hello, World!</h1>',
    #    }]
    #}
    return _metadata


def interactive_visualizer(data_set_path: str,
                           metadata_file_path: OutputPath(),
                           plot_type: str,
                           output_image_path: str,
                           title: str = None,
                           features: List[str] = None,
                           time_features: List[str] = None,
                           graph_features: List[str] = None,
                           group_by: List[str] = None,
                           analytical_data_types: Dict[str, List[str]] = None,
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
                           feature_tournament_game_stats: bool = False,
                           feature_tournament_game_size: bool = False,
                           feature_importance_shapley_scores: bool = False,
                           aggregate_feature_imp: Dict[str, dict] = None,
                           feature_importance_processing_variants: bool = False,
                           feature_importance_core_features_aggregation: bool = False,
                           sep: str = ','
                     ) -> NamedTuple('outputs', [('plot_file_path', str)]):
    """
    Interactive visualization

    :param data_set_path: str
        Complete file path of the data set

    :param analytical_data_types: dict
        Assigned analytical data types to each feature

    :param output_image_path: str
        Name of the output destination of the image

    :param sep: str
        Separator

    :return: NamedTuple
        Path of the engineered data set
    """
    _df: pd.DataFrame = pd.read_csv(filepath_or_buffer=data_set_path, sep=sep)
    _interactive_visualizer: InteractiveVisualizer = InteractiveVisualizer(df=_df,
                                                                           title=title,
                                                                           features=features,
                                                                           time_features=time_features,
                                                                           graph_features=graph_features,
                                                                           group_by=group_by,
                                                                           feature_types=analytical_data_types,
                                                                           plot_type=plot_type,
                                                                           subplots=None,
                                                                           melt=melt,
                                                                           brushing=brushing,
                                                                           xaxis_label=xaxis_label,
                                                                           yaxis_label=yaxis_label,
                                                                           zaxis_label=zaxis_label,
                                                                           annotations=annotations,
                                                                           width=width,
                                                                           height=height,
                                                                           unit=unit,
                                                                           file_path=output_image_path,
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
    _file_paths: List[str] = _interactive_visualizer.main(special_plots=False,
                                                          feature_tournament_game_stats=feature_tournament_game_stats,
                                                          feature_tournament_game_size=feature_tournament_game_size,
                                                          feature_importance_shapley_scores=feature_importance_shapley_scores,
                                                          aggregate_feature_imp=aggregate_feature_imp,
                                                          feature_importance_processing_variants=feature_importance_processing_variants,
                                                          feature_importance_core_features_aggregation=feature_importance_core_features_aggregation
                                                          )
    _kfp_metadata: Dict[str, List[Dict[str, str]]] = _generate_kfp_template(file_paths=_file_paths)
    with open(metadata_file_path, 'w') as metadata_file:
        json.dump(_kfp_metadata, metadata_file)
    return [_file_paths]


if __name__ == '__main__':
    interactive_visualizer(data_set_path=ARGS.data_set_path,
                           metadata_file_path='image',
                           plot_type=ARGS.plot_type,
                           output_image_path=ARGS.output_image_path,
                           title=ARGS.title,
                           features=ARGS.features,
                           time_features=ARGS.time_features,
                           graph_features=ARGS.graph_features,
                           group_by=ARGS.group_by,
                           analytical_data_types=ARGS.analytical_data_types,
                           subplots=ARGS.subplots,
                           melt=ARGS.melt,
                           brushing=ARGS.brushing,
                           xaxis_label=ARGS.xaxis_label,
                           yaxis_label=ARGS.yaxis_label,
                           zaxis_label=ARGS.zaxis_label,
                           annotations=ARGS.annotations,
                           width=ARGS.width,
                           height=ARGS.height,
                           unit=ARGS.unit,
                           use_auto_extensions=ARGS.use_auto_extensions,
                           color_scale=ARGS.color_scale,
                           color_edges=ARGS.color_edges,
                           color_feature=ARGS.color_feature,
                           feature_tournament_game_stats=ARGS.feature_tournament_game_stats,
                           feature_tournament_game_size=ARGS.feature_tournament_game_size,
                           feature_importance_shapley_scores=ARGS.feature_importance_shapley_scores,
                           aggregate_feature_imp=ARGS.aggregate_feature_imp,
                           feature_importance_processing_variants=ARGS.feature_importance_processing_variants,
                           feature_importance_core_features_aggregation=ARGS.feature_importance_core_features_aggregation,
                           sep=ARGS.sep,
                           )
