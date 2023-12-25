"""

Task: ... (Function to run in container)

"""

import argparse
import pandas as pd

from aws import load_file_from_s3, save_file_to_s3
from custom_logger import Log
from file_handler import file_handler
from feature_selector import FeatureSelector
from typing import List, NamedTuple

PARSER = argparse.ArgumentParser(description="feature selection")
PARSER.add_argument('-model_generator_path', type=str, required=True, default=None, help='complete file path of the model generator artifact')
PARSER.add_argument('-ml_type', type=str, required=True, default=None, help='name of the machine learning type')
PARSER.add_argument('-train_data_set_path', type=str, required=True, default=None, help='complete file path of the training data set')
PARSER.add_argument('-test_data_set_path', type=str, required=True, default=None, help='complete file path of the test data set')
PARSER.add_argument('-target_feature', type=str, required=True, default=None, help='name of the target feature')
PARSER.add_argument('-features', type=list, required=False, default=None, help='pre-defined feature names used for feature engineering')
PARSER.add_argument('-output_bucket_name', type=str, required=True, default=None, help='name of the S3 output bucket')
PARSER.add_argument('-output_file_path_data_set', type=str, required=True, default=None, help='file path of the data set')
PARSER.add_argument('-output_file_path_processor_obj', type=str, required=True, default=None, help='file path of output processor objects')
PARSER.add_argument('-output_file_path_target', type=str, required=True, default=None, help='file path of the output target feature')
PARSER.add_argument('-output_file_path_predictors', type=str, required=True, default=None, help='file path of the output predictors')
PARSER.add_argument('-output_file_path_engineered_feature_names', type=str, required=True, default=None, help='file path of the output processed features')
PARSER.add_argument('-sep', type=str, required=False, default=',', help='column separator')
ARGS = PARSER.parse_args()


def feature_selector(model_generator_path: str,
                     ml_type: str,
                     train_data_set_path: str,
                     test_data_set_path: str,
                     target_feature: str,
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
                     sep: str = ','
                     ) -> NamedTuple('outputs', [('file_path_data', str),
                                                 ('file_path_processor_obj', str),
                                                 ('engineered_feature_names', list),
                                                 ('predictors', list),
                                                 ('target', str)
                                                 ]
                                     ):
    """
    Feature selection of structured (tabular) data based on calculated feature importance scoring

    :param model_generator_path:
    :param ml_type:
    :param train_data_set_path:
    :param test_data_set_path:
    :param target_feature:
    :param features:
    :param init_pairs:
    :param init_games:
    :param increasing_pair_size_factor:
    :param games:
    :param penalty_factor:
    :param max_iter:
    :param max_players:
    :param imp_threshold:
    :param redundant_threshold:
    :param top_n_imp_features_proportion:
    :param feature_selection_algorithm:
    :param aggregate_feature_imp:
    :param sep:
    :return:
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
    for file_path, obj in [(output_path_metadata, _feature_selection.get('important')),
                           (output_path_metadata, _feature_selection.get('imp_features')),
                           (output_path_evaluation_data, _feature_selection.get('imp_score')),
                           (output_path_training_status, _feature_selection.get('imp_core_features')),
                           (output_path_evaluation_data, _feature_selection.get('imp_processed_features')),
                           (output_path_evaluation_data, _feature_selection.get('redundant')),
                           (output_path_evaluation_data, _feature_selection.get('gain')),
                           (output_path_evaluation_data, _feature_selection.get('model_metric')),
                           (output_path_evaluation_data, _feature_selection.get('base_metric')),
                           (output_path_evaluation_data, _feature_selection.get('threshold'))
                           ]:
        file_handler(file_path=file_path, obj=obj)
    if s3_output_path_visualization_data is not None:
        save_file_to_s3(file_path='', obj=_feature_selector.plot)
        Log().log(msg=f'Save visualization data: {s3_output_path_visualization_data}')
    return [output_file_path_data_set,
            output_file_path_processor_obj,
            _feature_names_engineered,
            _predictors,
            target_feature_name
            ]


if __name__ == '__main__':
    feature_selector(model_generator_path=ARGS.model_generator_path,
                     ml_type=ARGS.ml_type,
                     train_data_set_path=ARGS.train_data_set_path,
                     test_data_set_path=ARGS.test_data_set_path,
                     target_feature=ARGS.target_feature,
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
