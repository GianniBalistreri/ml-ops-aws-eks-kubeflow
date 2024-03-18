"""

Task: ... (Function to run in container)

"""

import argparse
import ast
import copy
import pandas as pd

from aws import load_file_from_s3, load_file_from_s3_as_df, save_file_to_s3, save_file_to_s3_as_df
from custom_logger import Log
from feature_engineering import FeatureEngineer
from file_handler import file_handler
from typing import Dict, List, NamedTuple

PARSER = argparse.ArgumentParser(description="feature engineering")
PARSER.add_argument('-data_set_path', type=str, required=True, default=None, help='file path of the data set')
PARSER.add_argument('-analytical_data_types_path', type=str, required=True, default=None, help='assignment of features to analytical data types')
PARSER.add_argument('-target_feature', type=str, required=True, default=None, help='name of the target feature')
PARSER.add_argument('-re_engineering', type=int, required=False, default=0, help='whether to re-engineer features for inference or not')
PARSER.add_argument('-next_level', type=int, required=False, default=0, help='whether to generate deeper engineered features or not')
PARSER.add_argument('-feature_engineering_config', type=str, required=False, default=None, help='feature engineering pre-defined config file')
PARSER.add_argument('-features', nargs='+', required=False, default=None, help='feature names used for feature engineering')
PARSER.add_argument('-ignore_features', nargs='+', required=False, default=None, help='pre-defined feature names to ignore in feature engineering')
PARSER.add_argument('-exclude_features', nargs='+', required=False, default=None, help='pre-defined feature names to exclude')
PARSER.add_argument('-exclude_original_data', type=int, required=False, default=0, help='whether to exclude all original features (especially numeric features) or not')
PARSER.add_argument('-exclude_meth', type=str, required=False, default=None, help='pre-defined feature engineering methods to exclude')
PARSER.add_argument('-use_only_meth', type=str, required=False, default=None, help='pre-defined feature engineering methods to use only')
PARSER.add_argument('-parallel_mode', type=int, required=False, default=0, help='whether to run task in parallel mode or not')
PARSER.add_argument('-sep', type=str, required=False, default=',', help='column separator')
PARSER.add_argument('-output_file_path_predictors', type=str, required=True, default=None, help='file path of the predictors output')
PARSER.add_argument('-output_file_path_new_target_feature', type=str, required=True, default=None, help='file path of the new target feature output')
PARSER.add_argument('-s3_output_file_path_data_set', type=str, required=True, default=None, help='S3 file path of the engineered data set')
PARSER.add_argument('-s3_output_file_path_processor_memory', type=str, required=True, default=None, help='S3 file path of the processor memory')
ARGS = PARSER.parse_args()


def feature_engineer(data_set_path: str,
                     analytical_data_types_path: str,
                     target_feature: str,
                     s3_output_file_path_data_set: str,
                     s3_output_file_path_processor_memory: str,
                     output_file_path_predictors: str,
                     output_file_path_new_target_feature: str,
                     re_engineering: bool = False,
                     next_level: bool = False,
                     feature_engineering_config: Dict[str, list] = None,
                     features: List[str] = None,
                     ignore_features: List[str] = None,
                     exclude_features: List[str] = None,
                     exclude_original_data: bool = False,
                     exclude_meth: List[str] = None,
                     use_only_meth: List[str] = None,
                     parallel_mode: bool = False,
                     sep: str = ',',
                     ) -> NamedTuple('outputs', [('predictors', list), ('new_target_feature', str)]):
    """
    Feature engineering of structured (tabular) data

    :param data_set_path: str
        Complete file path of the data set

    :param analytical_data_types_path: str
        Complete file path of the analytical data types

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

    :param re_engineering: bool
        Whether to re-engineer features for inference or to engineer for training

    :param next_level: bool
        Whether to engineer deeper (higher level) features or first level features

    :param feature_engineering_config: Dict[str, list]
            Pre-defined configuration

    :param features: List[str]
        Name of the features

    :param ignore_features: List[str]
        Name of the features to ignore in feature engineering

    :param exclude_features: List[str]
        Name of the features to exclude

    :param exclude_original_data: bool
        Exclude original features

    :param exclude_meth: List[str]
        Name of the feature engineering methods to exclude

    :param use_only_meth: List[str]
        Name of the feature engineering methods to use only

    :param parallel_mode: bool
        Whether to run task in parallel mode or not

    :param sep: str
        Separator

    :return: NamedTuple
        Features names, name of the (new) target feature
    """
    _analytical_data_types: Dict[str, List[str]] = load_file_from_s3(file_path=analytical_data_types_path)
    Log().log(msg=f'Load analytical data types: {analytical_data_types_path}')
    _df: pd.DataFrame = load_file_from_s3_as_df(file_path=data_set_path, sep=sep)
    Log().log(msg=f'Load data set: {data_set_path} -> Cases={_df.shape[0]}, Features={_df.shape[1]}')
    _features: List[str] = _df.columns.tolist() if features is None else features
    _target_feature: str = target_feature
    if _target_feature in _features:
        del _features[_features.index(_target_feature)]
    if exclude_features is not None:
        for feature in exclude_features:
            if feature in _features:
                del _features[_features.index(feature)]
                Log().log(msg=f'Exclude feature "{feature}"')
    _ignore_features: List[str] = []
    if ignore_features is not None:
        for feature in ignore_features:
            if feature in _features:
                _ignore_features.append(feature)
                del _features[_features.index(feature)]
                Log().log(msg=f'Ignore feature "{feature}"')
    _predictors: List[str] = copy.deepcopy(_features)
    if re_engineering:
        _feature_names_engineered: List[str] = None
        _processing_memory: dict = load_file_from_s3(file_path=s3_output_file_path_processor_memory)
        Log().log(msg=f'Load processing memory: {s3_output_file_path_processor_memory}')
        _feature_engineer: FeatureEngineer = FeatureEngineer(df=_df,
                                                             analytical_data_types=_analytical_data_types,
                                                             features=_features,
                                                             target_feature=_target_feature,
                                                             processing_memory=_processing_memory,
                                                             feature_engineering_config=feature_engineering_config,
                                                             exclude_meth=exclude_meth,
                                                             use_only_meth=use_only_meth
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
                                                             analytical_data_types=_analytical_data_types,
                                                             features=_features,
                                                             target_feature=_target_feature,
                                                             processing_memory=_processing_memory,
                                                             feature_engineering_config=_feature_engineering_config,
                                                             exclude_meth=exclude_meth,
                                                             use_only_meth=use_only_meth
                                                             )
        _df_engineered = _feature_engineer.main()
        Log().log(msg=f'Generated {_df_engineered.shape[1]} engineered features')
        _updated_analytical_data_types: Dict[str, List[str]] = _feature_engineer.processing_memory.get('analytical_data_types')
        _new_target_feature: str = _feature_engineer.processing_memory['new_target_feature']
        _feature_names_engineered: List[str] = _df_engineered.columns.tolist()
        _predictors.extend(_feature_names_engineered)
        if exclude_original_data:
            for raw in _features:
                if raw in _predictors:
                    del _predictors[_predictors.index(raw)]
                    Log().log(msg=f'Exclude original feature "{raw}"')
        else:
            for feature in _features:
                if feature in _predictors:
                    if feature not in _analytical_data_types.get('continuous'):
                        if feature not in _analytical_data_types.get('ordinal'):
                            del _predictors[_predictors.index(feature)]
                            Log().log(msg=f'Exclude original (non-numeric) feature "{feature}"')
        _predictors = sorted(_predictors)
        _processing_memory: dict = _feature_engineer.processing_memory
        _processing_memory.update({'predictors': _predictors})
        file_handler(file_path=output_file_path_predictors, obj=_predictors)
        file_handler(file_path=output_file_path_new_target_feature, obj=_new_target_feature)
        _df_output: pd.DataFrame = pd.concat(objs=[_df, _df_engineered], axis=1)
        _df_output = _df_output[_predictors]
        for feature in _ignore_features:
            _df_output[feature] = _df[feature].values
        if _target_feature in _df.columns.tolist():
            _df_output[_target_feature] = _df[_target_feature].values
        save_file_to_s3_as_df(file_path=s3_output_file_path_data_set, df=_df_output, sep=sep)
        Log().log(msg=f'Save engineered data set: {s3_output_file_path_data_set}')
        _analytical_data_types_path: str = analytical_data_types_path
        _s3_output_file_path_processor_memory: str = s3_output_file_path_processor_memory
        if parallel_mode:
            _suffix: str = data_set_path.split('.')[0].split('_')[-1]
            _analytical_data_types_path = analytical_data_types_path.replace('.', f'_{_suffix}.')
            _s3_output_file_path_processor_memory = s3_output_file_path_processor_memory.replace('.', f'_{_suffix}.')
        save_file_to_s3(file_path=_analytical_data_types_path, obj=_updated_analytical_data_types)
        Log().log(msg=f'Save updated analytical data types: {_analytical_data_types_path}')
        save_file_to_s3(file_path=_s3_output_file_path_processor_memory, obj=_processing_memory)
        Log().log(msg=f'Save processing memory: {_s3_output_file_path_processor_memory}')
    return [_predictors, _new_target_feature]


if __name__ == '__main__':
    if ARGS.features:
        ARGS.features = ast.literal_eval(ARGS.features[0])
    if ARGS.ignore_features:
        ARGS.ignore_features = ast.literal_eval(ARGS.ignore_features[0])
    if ARGS.feature_engineering_config:
        ARGS.feature_engineering_config = ast.literal_eval(ARGS.feature_engineering_config)
    if ARGS.exclude_meth:
        ARGS.exclude_meth = ast.literal_eval(ARGS.exclude_meth)
    if ARGS.use_only_meth:
        ARGS.use_only_meth = ast.literal_eval(ARGS.use_only_meth)
    feature_engineer(data_set_path=ARGS.data_set_path,
                     analytical_data_types_path=ARGS.analytical_data_types_path,
                     target_feature=ARGS.target_feature,
                     s3_output_file_path_data_set=ARGS.s3_output_file_path_data_set,
                     s3_output_file_path_processor_memory=ARGS.s3_output_file_path_processor_memory,
                     output_file_path_predictors=ARGS.output_file_path_predictors,
                     output_file_path_new_target_feature=ARGS.output_file_path_new_target_feature,
                     re_engineering=bool(ARGS.re_engineering),
                     next_level=bool(ARGS.next_level),
                     feature_engineering_config=ARGS.feature_engineering_config,
                     features=ARGS.features,
                     ignore_features=ARGS.ignore_features,
                     exclude_features=ARGS.exclude_features,
                     exclude_original_data=bool(ARGS.exclude_original_data),
                     exclude_meth=ARGS.exclude_meth,
                     use_only_meth=ARGS.use_only_meth,
                     parallel_mode=bool(ARGS.parallel_mode),
                     sep=ARGS.sep
                     )
