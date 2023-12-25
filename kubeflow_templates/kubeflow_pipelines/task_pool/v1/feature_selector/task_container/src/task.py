"""

Task: ... (Function to run in container)

"""

import argparse
import pandas as pd

from aws import load_file_from_s3, save_file_to_s3
from custom_logger import Log
from feature_selector import FeatureSelector
from file_handler import file_handler
from typing import Any, List, NamedTuple

PARSER = argparse.ArgumentParser(description="feature selection based on feature importance scoring using decision tree algorithms")
PARSER.add_argument('-model_generator_path', type=str, required=True, default=None, help='complete file path of the model generator artifact')
PARSER.add_argument('-ml_type', type=str, required=True, default=None, help='name of the machine learning type')
PARSER.add_argument('-train_data_set_path', type=str, required=True, default=None, help='complete file path of the training data set')
PARSER.add_argument('-test_data_set_path', type=str, required=True, default=None, help='complete file path of the test data set')
PARSER.add_argument('-target_feature', type=str, required=True, default=None, help='name of the target feature')
PARSER.add_argument('-features', type=list, required=False, default=None, help='pre-defined feature names used for feature engineering')
PARSER.add_argument('-init_pairs', type=int, required=False, default=3, help='number of feature pairs in the data set to start penalty initially')
PARSER.add_argument('-init_games', type=int, required=False, default=5, help='number of games each iteration to start penalty initially')
PARSER.add_argument('-increasing_pair_size_factor', type=float, required=False, default=0.5, help='increasing factor for gaining pair size each iteration')
PARSER.add_argument('-games', type=int, required=False, default=3, help='number of games each iteration to start tournament')
PARSER.add_argument('-penalty_factor', type=float, required=False, default=0.1, help='factor of removing features after penalty because of poor fitness')
PARSER.add_argument('-max_iter', type=int, required=False, default=50, help='maximum number of iterations of the tournament')
PARSER.add_argument('-max_players', type=int, required=False, default=-1, help='maximum number of players each game in the tournament')
PARSER.add_argument('-imp_threshold', type=float, required=False, default=0.01, help='')
PARSER.add_argument('-redundant_threshold', type=float, required=False, default=0.01, help='')
PARSER.add_argument('-top_n_imp_features_proportion', type=float, required=False, default=0.1, help='')
PARSER.add_argument('-feature_selection_algorithm', type=str, required=False, default='feature_addition', help='feature selection algorithm to apply')
PARSER.add_argument('-aggregate_feature_imp', type=Any, required=False, default=None, help='')
PARSER.add_argument('-sep', type=str, required=False, default=',', help='column separator')
PARSER.add_argument('-output_path_imp_features', type=str, required=True, default=None, help='file path of the output selected important features')
PARSER.add_argument('-output_path_ordered_features_by_imp_score', type=str, required=True, default=None, help='file path of the output features ordered by importance score')
PARSER.add_argument('-output_path_imp_scores', type=str, required=True, default=None, help='file path of the output importance score')
PARSER.add_argument('-output_path_imp_core_features', type=str, required=True, default=None, help='')
PARSER.add_argument('-output_path_imp_processed_features', type=str, required=True, default=None, help='file path of the output importance of processed features')
PARSER.add_argument('-output_path_redundant_features', type=str, required=True, default=None, help='file path of the output not selected redundant features')
PARSER.add_argument('-output_path_gain_scores', type=str, required=True, default=None, help='file path of the output model fitness gain score')
PARSER.add_argument('-output_path_reduction_scores', type=str, required=True, default=None, help='file path of the output model fitness reduction score')
PARSER.add_argument('-output_path_model_metric', type=str, required=True, default=None, help='file path of the output model metrics')
PARSER.add_argument('-output_path_base_model_metric', type=str, required=True, default=None, help='file path of the output base model metric')
PARSER.add_argument('-output_path_threshold_metric', type=str, required=True, default=None, help='file path of the output model threshold metric')
PARSER.add_argument('-s3_output_path_visualization_data', type=str, required=False, default=None, help='S3 file path of the visualization data output')
ARGS = PARSER.parse_args()


def feature_selector(model_generator_path: str,
                     ml_type: str,
                     train_data_set_path: str,
                     test_data_set_path: str,
                     target_feature: str,
                     output_path_imp_features: str,
                     output_path_ordered_features_by_imp_score: str,
                     output_path_imp_scores: str,
                     output_path_imp_core_features: str,
                     output_path_imp_processed_features: str,
                     output_path_redundant_features: str,
                     output_path_gain_scores: str,
                     output_path_reduction_scores: str,
                     output_path_model_metric: str,
                     output_path_base_model_metric: str,
                     output_path_threshold_metric: str,
                     features: List[str] = None,
                     init_pairs: int = 3,
                     init_games: int = 5,
                     increasing_pair_size_factor: float = 0.5,
                     games: int = 3,
                     penalty_factor: float = 0.1,
                     max_iter: int = 50,
                     max_players: int = -1,
                     imp_threshold: float = 0.01,
                     redundant_threshold: float = 0.01,
                     top_n_imp_features_proportion: float = 0.1,
                     feature_selection_algorithm: str = 'feature_addition',
                     aggregate_feature_imp: dict = None,
                     sep: str = ',',
                     s3_output_path_visualization_data: str = None,
                     ) -> NamedTuple('outputs', [('file_path_data', str),
                                                 ('file_path_processor_obj', str),
                                                 ('engineered_feature_names', list),
                                                 ('predictors', list),
                                                 ('target', str)
                                                 ]
                                     ):
    """
    Feature selection of structured (tabular) data based on calculated feature importance scoring

    :param model_generator_path: str
        Complete file path of the model generator artifact

    :param ml_type: str
        Abbreviated name of the machine learning type:
            ->

    :param train_data_set_path: str
        Complete file path of the training data set

    :param test_data_set_path: str
        Complete file path of the test data set

    :param target_feature: str
        Name of the target feature

    :param output_path_imp_features: str
        Output path of the important features

    :param output_path_ordered_features_by_imp_score: str
        Output path of the features ordered by importance score

    :param output_path_imp_scores: str
        Output path of the importance scores

    :param output_path_imp_core_features: str
        Output path of the

    :param output_path_imp_processed_features: str
        Output path of the importance of processed features

    :param output_path_redundant_features: str
        Output path of the redundant features

    :param output_path_gain_scores: str
        Output path of the model metric gain scores

    :param output_path_reduction_scores: str
        Output path of the model metric reduction scores

    :param output_path_model_metric: str
        Output path of the modl metrics

    :param output_path_base_model_metric: str
        Output path of the base model metric

    :param output_path_threshold_metric: str
        Output path of the model metric threshold

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

    :param aggregate_feature_imp: dict
        Relationship mapping of features to aggregate feature importance score

    :param sep: str
        Separator

    :param s3_output_path_visualization_data: str
        Complete file path of the visualization output

    :return: NamedTuple

    """
    _train_df: pd.DataFrame = pd.read_csv(filepath_or_buffer=train_data_set_path, sep=sep)
    _test_df: pd.DataFrame = pd.read_csv(filepath_or_buffer=test_data_set_path, sep=sep)
    _features: List[str] = _train_df.columns.tolist() if features is None else features
    _model_generator: object = load_file_from_s3(file_path=model_generator_path)
    _feature_selector: FeatureSelector = FeatureSelector(target_feature=target_feature,
                                                         features=_features,
                                                         train_df=_train_df,
                                                         test_df=_test_df,
                                                         model_generator=_model_generator,
                                                         ml_type=ml_type,
                                                         init_pairs=init_pairs,
                                                         init_games=init_games,
                                                         increasing_pair_size_factor=increasing_pair_size_factor,
                                                         games=games,
                                                         penalty_factor=penalty_factor,
                                                         max_iter=max_iter,
                                                         max_players=max_players,
                                                         redundant_threshold=redundant_threshold,
                                                         top_n_imp_features_proportion=top_n_imp_features_proportion
                                                         )
    _feature_selection: dict = _feature_selector.main(feature_selection_algorithm=feature_selection_algorithm,
                                                      imp_threshold=imp_threshold,
                                                      aggregate_feature_imp=aggregate_feature_imp
                                                      )
    for file_path, obj in [(output_path_imp_features, _feature_selection.get('important')),
                           (output_path_ordered_features_by_imp_score, _feature_selection.get('imp_features')),
                           (output_path_imp_scores, _feature_selection.get('imp_score')),
                           (output_path_imp_core_features, _feature_selection.get('imp_core_features')),
                           (output_path_imp_processed_features, _feature_selection.get('imp_processed_features')),
                           (output_path_redundant_features, _feature_selection.get('redundant')),
                           (output_path_gain_scores, _feature_selection.get('gain')),
                           (output_path_reduction_scores, _feature_selection.get('reduction')),
                           (output_path_model_metric, _feature_selection.get('model_metric')),
                           (output_path_base_model_metric, _feature_selection.get('base_metric')),
                           (output_path_threshold_metric, _feature_selection.get('threshold'))
                           ]:
        file_handler(file_path=file_path, obj=obj)
    if s3_output_path_visualization_data is not None:
        save_file_to_s3(file_path='', obj=_feature_selector.plot)
        Log().log(msg=f'Save visualization data: {s3_output_path_visualization_data}')
    return [_feature_selection.get('important'),
            _feature_selection.get('imp_features'),
            _feature_selection.get('imp_score'),
            _feature_selection.get('imp_core_features'),
            _feature_selection.get('imp_processed_features'),
            _feature_selection.get('redundant'),
            _feature_selection.get('gain'),
            _feature_selection.get('reduction'),
            _feature_selection.get('model_metric'),
            _feature_selection.get('base_metric'),
            _feature_selection.get('threshold')
            ]


if __name__ == '__main__':
    feature_selector(model_generator_path=ARGS.model_generator_path,
                     ml_type=ARGS.ml_type,
                     train_data_set_path=ARGS.train_data_set_path,
                     test_data_set_path=ARGS.test_data_set_path,
                     target_feature=ARGS.target_feature,
                     output_path_imp_features=ARGS.output_path_imp_features,
                     output_path_ordered_features_by_imp_score=ARGS.output_path_ordered_features_by_imp_score,
                     output_path_imp_scores=ARGS.output_path_imp_scores,
                     output_path_imp_core_features=ARGS.output_path_imp_core_features,
                     output_path_imp_processed_features=ARGS.output_path_imp_processed_features,
                     output_path_redundant_features=ARGS.output_path_redundant_features,
                     output_path_gain_scores=ARGS.output_path_gain_scores,
                     output_path_reduction_scores=ARGS.output_path_reduction_scores,
                     output_path_model_metric=ARGS.output_path_model_metric,
                     output_path_base_model_metric=ARGS.output_path_base_model_metric,
                     output_path_threshold_metric=ARGS.output_path_threshold_metric,
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
                     aggregate_feature_imp=ARGS.aggregate_feature_imp,
                     sep=ARGS.sep
                     )
