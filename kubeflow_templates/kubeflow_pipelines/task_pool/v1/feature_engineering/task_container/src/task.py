"""

Task: ... (Function to run in container)

"""

import argparse
import pandas as pd

from aws import load_file_from_s3, save_file_to_s3
from custom_logger import Log
from feature_engineering import FeatureEngineer
from file_handler import file_handler
from typing import Any, Dict, NamedTuple, List

PARSER = argparse.ArgumentParser(description="feature engineering")
PARSER.add_argument('-data_set_path', type=str, required=True, default=None, help='file path of the data set')
PARSER.add_argument('-analytical_data_types', type=Any, required=True, default=None, help='assignment of features to analytical data types')
PARSER.add_argument('-target_feature', type=str, required=True, default=None, help='name of the target feature')
PARSER.add_argument('-re_engineering', type=int, required=False, default=False, help='whether to re-engineer features for inference or not')
PARSER.add_argument('-next_level', type=int, required=False, default=False, help='whether to generate deeper engineered features or not')
PARSER.add_argument('-feature_engineering_config', type=Any, required=False, default=None, help='feature engineering pre-defined config file')
PARSER.add_argument('-features', type=str, required=False, default=None, help='feature names used for feature engineering')
PARSER.add_argument('-exclude', type=str, required=False, default=None, help='pre-defined feature names to exclude')
PARSER.add_argument('-exclude_original_data', type=int, required=False, default=0, help='whether to exclude all original features (especially numeric features) or not')
PARSER.add_argument('-sep', type=str, required=False, default=',', help='column separator')
PARSER.add_argument('-output_file_path_predictors', type=str, required=True, default=None, help='file path of the predictors output')
PARSER.add_argument('-output_file_path_new_target_feature', type=str, required=True, default=None, help='file path of the new target feature output')
PARSER.add_argument('-output_file_path_analytical_data_types', type=str, required=True, default=None, help='file path of the analytical data types output')
PARSER.add_argument('-s3_output_file_path_data_set', type=str, required=True, default=None, help='S3 file path of the engineered data set')
PARSER.add_argument('-s3_output_file_path_processor_memory', type=str, required=True, default=None, help='S3 file path of the processor memory')
ARGS = PARSER.parse_args()


def feature_engineer(data_set_path: str,
                     analytical_data_types: Dict[str, List[str]],
                     target_feature: str,
                     s3_output_file_path_data_set: str,
                     s3_output_file_path_processor_memory: str,
                     output_file_path_predictors: str,
                     output_file_path_new_target_feature: str,
                     output_file_path_analytical_data_types: str,
                     re_engineering: bool = False,
                     next_level: bool = False,
                     feature_engineering_config: Dict[str, list] = None,
                     features: List[str] = None,
                     exclude: List[str] = None,
                     exclude_original_data: bool = False,
                     sep: str = ','
                     ) -> NamedTuple('outputs', [('predictors', list),
                                                 ('new_target_feature', str),
                                                 ('analytical_data_types', dict)
                                                 ]
                                     ):
    """
    Feature engineering of structured (tabular) data

    :param data_set_path: str
        Complete file path of the data set

    :param analytical_data_types: dict
        Assigned analytical data types to each feature

    :param target_feature: str
        Name of the target feature

    :param s3_output_file_path_data_set: str
        Complete file path of the data set to save

    :param s3_output_file_path_processor_memory: str
        Complete file path of the processing memory to save

    :param output_file_path_predictors: str
        Path of the predictors output

    :param output_file_path_new_target_feature: str
        Path of the new target feature output

    :param output_file_path_analytical_data_types: str
        Path of the updated analytical data types output

    :param re_engineering: bool
        Whether to re-engineer features for inference or to engineer for training

    :param next_level: bool
        Whether to engineer deeper (higher level) features or first level features

    :param feature_engineering_config: Dict[str, list]
            Pre-defined configuration

    :param features: List[str]
        Name of the features

    :param exclude: List[str]
        Name of the features to exclude

    :param exclude_original_data: bool
        Exclude original features

    :param sep: str
        Separator

    :return: NamedTuple
        Path of the engineered data set
    """
    _df: pd.DataFrame = pd.read_csv(filepath_or_buffer=data_set_path, sep=sep)
    _features: List[str] = _df.columns.tolist() if features is None else features
    _target_feature: str = target_feature
    if _target_feature in _features:
        del _features[_features.index(_target_feature)]
    _predictors: List[str] = _features
    if re_engineering:
        _feature_names_engineered: List[str] = None
        _processing_memory: dict = load_file_from_s3(file_path=s3_output_file_path_processor_memory)
        _feature_engineer: FeatureEngineer = FeatureEngineer(df=_df,
                                                             analytical_data_types=analytical_data_types,
                                                             features=_features,
                                                             target_feature=_target_feature,
                                                             processing_memory=_processing_memory,
                                                             feature_engineering_config=feature_engineering_config
                                                             )
        _df_engineered = _feature_engineer.re_engineering(features=_features)
        _updated_analytical_data_types: Dict[str, List[str]] = _feature_engineer.processing_memory.get('analytical_data_types')
        _new_target_feature: str = _target_feature
    else:
        if next_level:
            _processing_memory: dict = load_file_from_s3(file_path=s3_output_file_path_processor_memory)
        else:
            _processing_memory: dict = None
        _feature_engineering_config: Dict[str, list] = feature_engineering_config
        _feature_engineer: FeatureEngineer = FeatureEngineer(df=_df,
                                                             analytical_data_types=analytical_data_types,
                                                             features=_features,
                                                             target_feature=_target_feature,
                                                             processing_memory=_processing_memory,
                                                             feature_engineering_config=_feature_engineering_config
                                                             )
        _df_engineered = _feature_engineer.main()
        _updated_analytical_data_types: Dict[str, List[str]] = _feature_engineer.processing_memory.get('analytical_data_types')
        _new_target_feature: str = _feature_engineer.processing_memory['new_target_feature']
        pd.concat(objs=[_df, _df_engineered], axis=1).to_csv(path_or_buf=s3_output_file_path_data_set, sep=sep, header=True, index=False)
        Log().log(msg=f'Save engineered data set: {s3_output_file_path_data_set}')
        _feature_names_engineered: List[str] = _df_engineered.columns.tolist()
        _predictors.extend(_feature_names_engineered)
        if exclude is not None:
            for feature in exclude:
                if feature in _predictors:
                    del _predictors[_predictors.index(feature)]
                    Log().log(msg=f'Exclude feature "{feature}"')
        if exclude_original_data:
            for raw in _features:
                if raw in _predictors:
                    del _predictors[_predictors.index(raw)]
                    Log().log(msg=f'Exclude original feature "{raw}"')
        else:
            for feature in _features:
                if feature in _predictors:
                    if feature not in analytical_data_types.get('continuous'):
                        if feature not in analytical_data_types.get('ordinal'):
                            del _predictors[_predictors.index(feature)]
                            Log().log(msg=f'Exclude original (non-numeric) feature "{feature}"')
        for non_numeric in _feature_engineer.processing_memory.get('exclude'):
            if non_numeric in _predictors:
                del _predictors[_predictors.index(non_numeric)]
                Log().log(msg=f'Exclude original (non-numeric) feature "{non_numeric}"')
        _predictors = sorted(_predictors)
    for file_path, obj in [(output_file_path_predictors, _predictors),
                           (output_file_path_new_target_feature, _new_target_feature),
                           (output_file_path_analytical_data_types, _updated_analytical_data_types)
                           ]:
        file_handler(file_path=file_path, obj=obj)
    save_file_to_s3(file_path=s3_output_file_path_processor_memory, obj=_feature_engineer.processing_memory)
    Log().log(msg=f'Save processing memory: {s3_output_file_path_processor_memory}')
    return [_predictors, _new_target_feature, _updated_analytical_data_types]


if __name__ == '__main__':
    feature_engineer(data_set_path=ARGS.data_set_path,
                     analytical_data_types=ARGS.analytical_data_types,
                     target_feature=ARGS.target_feature_name,
                     s3_output_file_path_data_set=ARGS.s3_output_file_path_data_set,
                     s3_output_file_path_processor_memory=ARGS.s3_output_file_path_processor_memory,
                     output_file_path_predictors=ARGS.output_file_path_predictors,
                     output_file_path_new_target_feature=ARGS.output_file_path_new_target_feature,
                     output_file_path_analytical_data_types=ARGS.output_file_path_analytical_data_types,
                     re_engineering=ARGS.re_engineering,
                     next_level=ARGS.next_level,
                     feature_engineering_config=ARGS.feature_engineering_config,
                     features=ARGS.features,
                     exclude=ARGS.exclude,
                     exclude_original_data=bool(ARGS.exclude_original_data),
                     sep=ARGS.sep
                     )
