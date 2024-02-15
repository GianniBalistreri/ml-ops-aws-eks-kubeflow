"""

Task: ... (Function to run in container)

"""

import argparse
import ast
import pandas as pd

from aws import load_file_from_s3, load_file_from_s3_as_df, save_file_to_s3
from custom_logger import Log
from feature_selector import FeatureSelector
from file_handler import file_handler
from typing import List, NamedTuple

PARSER = argparse.ArgumentParser(description="feature selection based on feature importance scoring using decision tree algorithms")
PARSER.add_argument('-ml_type', type=str, required=True, default=None, help='name of the machine learning type')
PARSER.add_argument('-train_data_set_path', type=str, required=True, default=None, help='complete file path of the training data set')
PARSER.add_argument('-test_data_set_path', type=str, required=True, default=None, help='complete file path of the test data set')
PARSER.add_argument('-target_feature', type=str, required=True, default=None, help='name of the target feature')
PARSER.add_argument('-features', nargs='+', required=False, default=None, help='pre-defined feature names used for feature engineering')
PARSER.add_argument('-init_pairs', type=int, required=False, default=3, help='number of feature pairs in the data set to start penalty initially')
PARSER.add_argument('-init_games', type=int, required=False, default=5, help='number of games each iteration to start penalty initially')
PARSER.add_argument('-increasing_pair_size_factor', type=float, required=False, default=0.05, help='increasing factor for gaining pair size each iteration')
PARSER.add_argument('-games', type=int, required=False, default=3, help='number of games each iteration to start tournament')
PARSER.add_argument('-penalty_factor', type=float, required=False, default=0.1, help='factor of removing features after penalty because of poor fitness')
PARSER.add_argument('-max_iter', type=int, required=False, default=50, help='maximum number of iterations of the tournament')
PARSER.add_argument('-max_players', type=int, required=False, default=-1, help='maximum number of players each game in the tournament')
PARSER.add_argument('-imp_threshold', type=float, required=False, default=0.01, help='')
PARSER.add_argument('-redundant_threshold', type=float, required=False, default=0.01, help='')
PARSER.add_argument('-top_n_imp_features_proportion', type=float, required=False, default=0.1, help='')
PARSER.add_argument('-feature_selection_algorithm', type=str, required=False, default='feature_addition', help='feature selection algorithm to apply')
PARSER.add_argument('-feature_selection_early_stopping', type=int, required=False, default=0, help='whether to stop early if hit redundant threshold applying feature selection algorithm')
PARSER.add_argument('-model_name', type=str, required=False, default='xgb', help='abbreviated name of the supervised machine learning model')
PARSER.add_argument('-model_param_path', type=str, required=False, default=None, help='complete file path of the pre-defined model hyperparameter')
PARSER.add_argument('-aggregate_feature_imp', type=str, required=False, default=None, help='')
PARSER.add_argument('-sep', type=str, required=False, default=',', help='column separator')
PARSER.add_argument('-output_path_imp_features', type=str, required=True, default=None, help='file path of the output selected important features')
PARSER.add_argument('-s3_output_path_metadata', type=str, required=False, default=None, help='S3 file path of the visualization data output')
PARSER.add_argument('-s3_output_path_visualization_data', type=str, required=False, default=None, help='S3 file path of the visualization data output')
ARGS = PARSER.parse_args()


def feature_selector(ml_type: str,
                     train_data_set_path: str,
                     test_data_set_path: str,
                     target_feature: str,
                     output_path_imp_features: str,
                     features: List[str] = None,
                     init_pairs: int = 3,
                     init_games: int = 5,
                     increasing_pair_size_factor: float = 0.05,
                     games: int = 3,
                     penalty_factor: float = 0.1,
                     max_iter: int = 50,
                     max_players: int = -1,
                     imp_threshold: float = 0.01,
                     redundant_threshold: float = 0.01,
                     top_n_imp_features_proportion: float = 0.1,
                     feature_selection_algorithm: str = 'feature_addition',
                     feature_selection_early_stopping: bool = False,
                     model_name: str = 'xgb',
                     model_param_path: str = None,
                     aggregate_feature_imp: dict = None,
                     sep: str = ',',
                     s3_output_path_metadata: str = None,
                     s3_output_path_visualization_data: str = None,
                     ) -> NamedTuple('outputs', [('imp_predictors', list)]):
    """
    Feature selection of structured (tabular) data based on calculated feature importance scoring

    :param ml_type: str
        Abbreviated name of the machine learning type:
            -> reg: Regression
            -> clf_binary: Binary Classification
            -> clf_multi: Multi-Classification

    :param train_data_set_path: str
        Complete file path of the training data set

    :param test_data_set_path: str
        Complete file path of the test data set

    :param target_feature: str
        Name of the target feature

    :param output_path_imp_features: str
        Output path of the important features

    :param features: List[str]
        Name of the features

    :param init_pairs: int
        Number of feature pairs in initial penalty

    :param init_games: int
        Number of the games each interation in initial penalty

    :param increasing_pair_size_factor: float
        Factor of increasing number of feature pairs each iteration during tournament

    :param games: int
        Number of games each iteration of the tournament

    :param penalty_factor: float
        Factor of removing features during penalty because of poor contribution to fitness metric

    :param max_iter: int
        Maximum number of iterations during tournament

    :param max_players: int
        Maximum number of players during tournament

    :param imp_threshold: float

    :param redundant_threshold: float
        Threshold to decide whether a feature is redundant or not

    :param top_n_imp_features_proportion: float
        Proportion of top n features to select a feature used in filter-based method

    :param feature_selection_algorithm: str
        Name of the feature selection algorithm to use
            -> feature_addition: Feature addition
            -> filter-based: Filter-based
            -> recursive_feature_elimination: Recursive feature elimination (RFE)

    :param feature_selection_early_stopping: bool
        Whether to stop early if hit redundant threshold applying feature selection algorithm

    :param model_name: str
        Abbreviated name of the supervised machine learning model
            -> cat: CatBoost
            -> gbo: Gradient Boosting Decision Tree
            -> rf: Random Forest
            -> xgb: Extreme Gradient Boosting Decision Tree

    :param model_param_path: str
        Complete file path of the pre-defined model hyperparameter

    :param aggregate_feature_imp: dict
        Relationship mapping of features to aggregate feature importance score

    :param sep: str
        Separator

    :param s3_output_path_metadata: str
        Complete file path of the metadata output

    :param s3_output_path_visualization_data: str
        Complete file path of the visualization output

    :return: NamedTuple
        Selected most important predictors
    """
    _train_df: pd.DataFrame = load_file_from_s3_as_df(file_path=train_data_set_path, sep=sep)
    Log().log(msg=f'Load training data set: {train_data_set_path} -> Cases={_train_df.shape[0]}, Features={_train_df.shape[1]}')
    _test_df: pd.DataFrame = load_file_from_s3_as_df(file_path=test_data_set_path, sep=sep)
    Log().log(msg=f'Load test data set: {test_data_set_path} -> Cases={_test_df.shape[0]}, Features={_test_df.shape[1]}')
    _features: List[str] = _train_df.columns.tolist() if features is None else features
    if target_feature in _features:
        del _features[_features.index(target_feature)]
    if model_param_path is None:
        _model_param: dict = None
    else:
        _model_param: dict = load_file_from_s3(file_path=model_param_path)
    _feature_selector: FeatureSelector = FeatureSelector(target_feature=target_feature,
                                                         features=_features,
                                                         train_df=_train_df,
                                                         test_df=_test_df,
                                                         ml_type=ml_type,
                                                         init_pairs=init_pairs,
                                                         init_games=init_games,
                                                         increasing_pair_size_factor=increasing_pair_size_factor,
                                                         games=games,
                                                         penalty_factor=penalty_factor,
                                                         max_iter=max_iter,
                                                         max_players=max_players,
                                                         feature_selection_early_stopping=feature_selection_early_stopping,
                                                         redundant_threshold=redundant_threshold,
                                                         top_n_imp_features_proportion=top_n_imp_features_proportion,
                                                         model_name=model_name,
                                                         model_param=None
                                                         )
    _file_name_subplots: str = s3_output_path_visualization_data.split('/')[-1]
    _file_path_plot: str = s3_output_path_visualization_data.replace(_file_name_subplots, '')
    _feature_selection: dict = _feature_selector.main(feature_selection_algorithm=feature_selection_algorithm,
                                                      imp_threshold=imp_threshold,
                                                      aggregate_feature_imp=aggregate_feature_imp,
                                                      plot_path=_file_path_plot
                                                      )
    file_handler(file_path=output_path_imp_features, obj=_feature_selection.get('important'))
    if s3_output_path_metadata is not None:
        save_file_to_s3(file_path=s3_output_path_metadata, obj=_feature_selection)
        Log().log(msg=f'Save metadata: {s3_output_path_metadata}')
    if s3_output_path_visualization_data is not None:
        save_file_to_s3(file_path=s3_output_path_visualization_data, obj=_feature_selector.plot)
        Log().log(msg=f'Save visualization data: {s3_output_path_visualization_data}')
    return [_feature_selection.get('important')]


if __name__ == '__main__':
    if ARGS.features:
        ARGS.features = ast.literal_eval(ARGS.features[0])
    if ARGS.aggregate_feature_imp:
        ARGS.aggregate_feature_imp = ast.literal_eval(ARGS.aggregate_feature_imp)
    feature_selector(ml_type=ARGS.ml_type,
                     train_data_set_path=ARGS.train_data_set_path,
                     test_data_set_path=ARGS.test_data_set_path,
                     target_feature=ARGS.target_feature,
                     output_path_imp_features=ARGS.output_path_imp_features,
                     features=ARGS.features,
                     init_pairs=ARGS.init_pairs,
                     init_games=ARGS.init_games,
                     increasing_pair_size_factor=ARGS.increasing_pair_size_factor,
                     games=ARGS.games,
                     penalty_factor=ARGS.penalty_factor,
                     max_iter=ARGS.max_iter,
                     imp_threshold=ARGS.imp_threshold,
                     redundant_threshold=ARGS.redundant_threshold,
                     top_n_imp_features_proportion=ARGS.top_n_imp_features_proportion,
                     feature_selection_algorithm=ARGS.feature_selection_algorithm,
                     feature_selection_early_stopping=bool(ARGS.feature_selection_early_stopping),
                     model_name=ARGS.model_name,
                     model_param_path=ARGS.model_param_path,
                     aggregate_feature_imp=ARGS.aggregate_feature_imp,
                     sep=ARGS.sep,
                     s3_output_path_metadata=ARGS.s3_output_path_metadata,
                     s3_output_path_visualization_data=ARGS.s3_output_path_visualization_data
                     )
