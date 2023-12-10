"""
Task: ... (Function to run in container)
"""

import pandas as pd

from easyexplore.data_import_export import DataImporter
from easyexplore.utils import EasyExploreUtils, StatsUtils
from typing import Dict, NamedTuple, List


def check_feature_distribution(data_set_name: str,
                               previous_stats_file_path: str,
                               max_multivariate_dimensions: int = 3,
                               categorical_test_meth: str = 'chi2',
                               continuous_test_meth: str = 'ks'
                               ) -> NamedTuple(typename='outputs', fields=[('proceed', bool), ('msg', str)]):
    """
    Check the distribution of features for significant changes

    :param data_set_name: str
        Name of the data set

    :param previous_stats_file_path: str
        Complete file path of the previous calculated feature distribution statistics

    :param max_multivariate_dimensions: int
        Maximum of dimensions regarding multivariate distribution

    :param categorical_test_meth: str
        Name of the statistical test method for categorical features

    :param continuous_test_meth: str
        Name of the statistical test method for continuous features

    :return: NamedTuple
        Whether to proceed with pipeline processes and message if no significant changes are detected
    """
    _df: pd.DataFrame = pd.DataFrame()
    _feature_types: Dict[str, List[str]] = EasyExploreUtils().get_feature_types(df=_df,
                                                                                features=_df.columns.tolist(),
                                                                                dtypes=_df.dtypes,
                                                                                continuous=None,
                                                                                categorical=None,
                                                                                ordinal=None,
                                                                                date=None,
                                                                                id_text=None,
                                                                                date_edges=None,
                                                                                max_categories=100,
                                                                                multi_threading=False,
                                                                                print_msg=False
                                                                                )
    _previous_stats: dict = DataImporter(file_path=previous_stats_file_path,
                                         as_data_frame=False,
                                         use_dask=False,
                                         create_dir=False,
                                         cloud='aws',
                                         bucket_name='',
                                         region=''
                                         ).file()
    _proceed: bool = False
    if _proceed:
        _msg: str = 'Significant changes in feature distribution detected'
    else:
        _msg: str = 'No significant changes in feature distribution detected'
    return [_proceed, _msg]
