"""

Generate interactive visualization using a plot.ly (offline API) wrapper

"""

import pandas as pd

from data_visualizer import DataVisualizer
from typing import Dict, List


class InteractiveVisualizerException(Exception):
    """
    Class for handling exceptions for class InteractiveVisualizer
    """
    pass


class InteractiveVisualizer:
    """
    Class for generating interactive plot.ly visualizations
    """
    def __init__(self,
                 df: pd.DataFrame,
                 title: str = '',
                 features: List[str] = None,
                 time_features: List[str] = None,
                 graph_features: Dict[str, str] = None,
                 group_by: List[str] = None,
                 feature_types: Dict[str, List[str]] = None,
                 plot_type: str = None,
                 subplots: dict = None,
                 melt: bool = False,
                 brushing: bool = True,
                 xaxis_label: List[str] = None,
                 yaxis_label: List[str] = None,
                 zaxis_label: List[str] = None,
                 annotations: List[dict] = None,
                 width: int = 500,
                 height: int = 500,
                 unit: str = 'px',
                 file_path: str = None,
                 use_auto_extensions: bool = False,
                 cloud: str = None,
                 render: bool = False,
                 color_scale: List[str] = None,
                 color_edges: List[str] = None,
                 color_feature: str = None,
                 max_row: int = 50,
                 max_col: int = 20,
                 rows_sub: int = None,
                 cols_sub: int = None
                 ):
        """
        :param df: pd.DataFrame
            Training data set

        :param features: List[str]
            Name of the features

        :param features: List[str]
            Name of the features

        :param time_features: List[str]
            Name of the time regarding features in line plots

        :param feature_types: Dict[str, List[str]]
            Pre-defined feature type segmentation

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

        :param melt: bool
            Melt subplots into one main plot

        :param brushing: bool
            Generate additional scatter chart for case-based exploration of feature connections

        :param xaxis_label: List[str]
            User based labeling of the xaxis for all subplots

        :param yaxis_label: List[str]
            User based labeling of the yaxis for all subplots

        :param annotations: List[dict]
            Annotation configuration for each subplot

        :param subplots: dict
            Subplot configuration

        :param rows_sub: int
            Number of rows to use in subplot

        :param cols_sub: int
            Number of columns to use in subplot

        :param width: int
            Width size for each subplot

        :param height: int
            Height size for each subplot

        :param unit: str
            Measurement unit
                -> px, pixel: Pixel
                -> in, inch: Inch
                -> cm, centimeter: Centimeter

        :param file_path: str
            File path of the plot to save

        :param use_auto_extensions: bool
            Use automatic file name extensions beyond group by functionality

        :param cloud: str
            Abbreviated name of the cloud provider
                -> aws: Amazon Web Services
                -> google: Google Cloud Platform

        :param render: bool
            Render plotly chart or not

        :param max_row: int
            Maximum number of rows of visualized Pandas DataFrames

        :param max_col: int
            Maximum number of columns of visualized Pandas DataFrames
        """
        self.title: str = title
        self.df: pd.DataFrame = df
        self.features: List[str] = self.df.columns.tolist() if features is None else features
        self.n_features: int = len(self.features)
        self.time_features: List[str] = time_features
        self.graph_features: Dict[str, str] = graph_features
        self.group_by: List[str] = group_by
        self.feature_types: Dict[str, List[str]] = feature_types
        self.plot_type: str = plot_type
        self.subplots: dict = subplots
        self.melt: bool = melt
        self.brushing: bool = brushing
        self.xaxis_label: List[str] = xaxis_label
        self.yaxis_label: List[str] = yaxis_label
        self.zaxis_label: List[str] = zaxis_label
        self.annotations: List[dict] = annotations
        self.width: int = width
        self.height: int = height
        self.unit: str = unit
        self.file_path: str = file_path.replace('\\', '/')
        self.path: str = self.file_path.split('.')[0]
        self.use_auto_extensions: bool = use_auto_extensions
        self.cloud: str = cloud
        self.render: bool = render
        self.color_scale: List[str] = color_scale
        self.color_edges: List[str] = color_edges
        self.color_feature: str = color_feature
        self.max_row: int = max_row
        self.max_col: int = max_col
        self.rows_sub: int = rows_sub
        self.cols_sub: int = cols_sub

    def main(self) -> List[str]:
        """
        Generate interactive visualization

        :return List[str]
            File paths of the generated and persisted interactive visualizations
        """
        _data_visualizer: DataVisualizer = DataVisualizer(df=self.df,
                                                          title=self.title,
                                                          features=self.features,
                                                          time_features=self.time_features,
                                                          graph_features=self.graph_features,
                                                          group_by=self.group_by,
                                                          feature_types=self.feature_types,
                                                          plot_type=self.plot_type,
                                                          subplots=self.subplots,
                                                          melt=self.melt,
                                                          brushing=self.brushing,
                                                          xaxis_label=self.xaxis_label,
                                                          yaxis_label=self.yaxis_label,
                                                          zaxis_label=self.zaxis_label,
                                                          annotations=self.annotations,
                                                          width=self.width,
                                                          height=self.height,
                                                          unit=self.unit,
                                                          interactive=True,
                                                          file_path=self.file_path,
                                                          use_auto_extensions=self.use_auto_extensions,
                                                          cloud=self.cloud,
                                                          render=self.render,
                                                          color_scale=self.color_scale,
                                                          color_edges=self.color_edges,
                                                          color_feature=self.color_feature,
                                                          max_row=self.max_row,
                                                          max_col=self.max_col,
                                                          rows_sub=self.rows_sub,
                                                          cols_sub=self.cols_sub
                                                          )
        _data_visualizer.run()
        return _data_visualizer.file_paths
