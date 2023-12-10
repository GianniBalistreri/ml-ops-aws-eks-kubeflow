"""

Task: ... (Function to run in container)

"""

import argparse
import boto3
import pandas as pd

from aws import load_file_from_s3, save_file_to_s3
from custom_logger import Log
from feature_engineering import ENGINEERING_METH, MIN_FEATURES_BY_METH, FeatureEngineer
from file_handler import file_handler
from typing import Any, Dict, NamedTuple, List

PARSER = argparse.ArgumentParser(description="feature engineering")
PARSER.add_argument('-data_set_path', type=str, required=True, default=None, help='file path of the data set')
PARSER.add_argument('-analytical_data_types', type=Any, required=True, default=None, help='assignment of features to analytical data types')
PARSER.add_argument('-target_feature_name', type=str, required=True, default=None, help='name of the target feature')
PARSER.add_argument('-output_bucket_name', type=str, required=True, default=None, help='name of the S3 output bucket')
PARSER.add_argument('-output_file_path_data_set', type=str, required=True, default=None, help='file path of the data set')
PARSER.add_argument('-output_file_path_processor_memory', type=str, required=True, default=None, help='file path of output processing memory')
PARSER.add_argument('-output_file_path_target', type=str, required=True, default=None, help='file path of the output target feature')
PARSER.add_argument('-output_file_path_predictors', type=str, required=True, default=None, help='file path of the output predictors')
PARSER.add_argument('-output_file_path_engineered_feature_names', type=str, required=True, default=None, help='file path of the output processed features')
PARSER.add_argument('-re_engineering', type=bool, required=False, default=False, help='whether to re-engineer features for inference or not')
PARSER.add_argument('-next_level', type=bool, required=False, default=False, help='whether to generate deeper engineered features or not')
PARSER.add_argument('-feature_engineering_config', type=Any, required=False, default=None, help='feature engineering pre-defined config file')
PARSER.add_argument('-feature_names', type=str, required=False, default=None, help='pre-defined feature names used for feature engineering')
PARSER.add_argument('-exclude', type=str, required=False, default=None, help='pre-defined feature names to exclude')
PARSER.add_argument('-exclude_original_data', type=str, required=False, default=None, help='exclude all original features (especially numeric features)')
PARSER.add_argument('-sep', type=str, required=False, default=',', help='column separator')
ARGS = PARSER.parse_args()


def feature_engineer(data_set_path: str,
                     analytical_data_types: dict,
                     target_feature_name: str,
                     output_file_path_data_set: str,
                     output_file_path_processor_memory: str,
                     output_file_path_target: str,
                     output_file_path_predictors: str,
                     output_file_path_engineered_feature_names: str,
                     re_engineering: bool = False,
                     next_level: bool = False,
                     feature_engineering_config: Dict[str, list] = None,
                     feature_names: List[str] = None,
                     exclude: List[str] = None,
                     exclude_original_data: bool = False,
                     sep: str = ',',
                     output_file_path_data_set_customized: str = None,
                     output_file_path_processor_memory_customized: str = None,
                     output_file_path_target_customized: str = None,
                     output_file_path_predictors_customized: str = None,
                     output_file_path_engineered_feature_names_customized: str = None
                     ) -> NamedTuple('outputs', [('file_path_data', str),
                                                 ('file_path_processor_obj', str),
                                                 ('engineered_feature_names', list),
                                                 ('predictors', list),
                                                 ('target', str)
                                                 ]
                                     ):
    """
    Feature engineering of structured (tabular) data

    :param data_set_path: str
        Complete file path of the data set

    :param analytical_data_types: dict
        Assigned analytical data types to each feature

    :param target_feature_name: str
        Name of the target feature

    :param output_file_path_data_set: str
        Path of the data set to save

    :param output_file_path_processor_memory: str
        Path of the processing memory to save

    :param output_file_path_target: str
        Path of the target feature to save

    :param output_file_path_predictors: str
        Path of the predictors to save

    :param output_file_path_engineered_feature_names: str
        Path of the engineered feature names

    :param re_engineering: bool
        Whether to re-engineer features for inference or to engineer for training

    :param next_level: bool
        Whether to engineer deeper (higher level) features or first level features

    :param feature_engineering_config: Dict[str, list]
            Pre-defined configuration

    :param feature_names: List[str]
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
    _features: List[str] = _df.columns.tolist() if feature_names is None else feature_names
    _s3_resource: boto3 = boto3.resource('s3')
    if re_engineering:
        _predictors: List[str] = _features
        _feature_names_engineered: List[str] = None
        _processing_memory: dict = load_file_from_s3(file_path=output_file_path_processor_memory,
                                                     encoding='utf-8'
                                                     )
        _feature_engineer: FeatureEngineer = FeatureEngineer(df=_df, processing_memory=_processing_memory)
        _df_engineered = _feature_engineer.re_engineering(features=_predictors)
    else:
        if next_level:
            _processing_memory: dict = load_file_from_s3(file_path=output_file_path_processor_memory,
                                                         encoding='utf-8'
                                                         )
            _feature_engineering_config: Dict[str, list] = {}
        else:
            _processing_memory: dict = None
            if feature_engineering_config is None:
                _feature_engineering_config: Dict[str, list] = {}
                for analytical_data_type in analytical_data_types.keys():
                    _n_features: int = len(analytical_data_types.get(analytical_data_type))
                    if _n_features > 0:
                        for meth in ENGINEERING_METH.get(analytical_data_type):
                            _min_features: int = 1 if MIN_FEATURES_BY_METH.get(meth) is None else MIN_FEATURES_BY_METH.get(meth)
                            if _n_features >= _min_features:
                                _feature_engineering_config.update({meth: []})
                                for feature in analytical_data_types.get(analytical_data_type):
                                    if _min_features == 1:
                                        if feature in _features:
                                            _feature_engineering_config[meth].append(feature)
                                    else:
                                        for interactor in analytical_data_types.get(analytical_data_type):
                                            if feature in _features:
                                                if feature != interactor:
                                                    if interactor in _features:
                                                        _feature_engineering_config[meth].append((feature, interactor))
            else:
                _feature_engineering_config: Dict[str, list] = feature_engineering_config
        _feature_engineer: FeatureEngineer = FeatureEngineer(df=_df, processing_memory=_processing_memory)
        _df_engineered = _feature_engineer.main(feature_engineering_config=_feature_engineering_config)
        _file_path: str = output_file_path_data_set if output_file_path_processor_memory_customized.find('s3://') >= 0 else f's3://{output_file_path_data_set}'
        pd.concat(objs=[_df, _df_engineered], axis=1).to_csv(path_or_buf=_file_path, sep=sep, index=False)
        _feature_names_engineered: List[str] = _df_engineered.columns.tolist()
        _predictors: List[str] = _features
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
    for file_path, obj, customized_file_path in [(output_file_path_data_set, output_file_path_data_set, output_file_path_data_set_customized),
                                                 (output_file_path_target, target_feature_name, output_file_path_target_customized),
                                                 (output_file_path_predictors, _predictors, output_file_path_predictors_customized),
                                                 (output_file_path_engineered_feature_names, _feature_names_engineered, output_file_path_engineered_feature_names_customized)
                                                 ]:
        file_handler(file_path=file_path, obj=obj)
        if customized_file_path is not None:
            save_file_to_s3(file_path=customized_file_path, obj=obj)
    return [output_file_path_data_set,
            output_file_path_processor_memory,
            _feature_names_engineered,
            _predictors,
            target_feature_name
            ]


if __name__ == '__main__':
    feature_engineer(data_set_path=ARGS.data_set_path,
                     analytical_data_types=ARGS.analytical_data_types,
                     target_feature_name=ARGS.target_feature_name,
                     output_file_path_data_set=ARGS.output_file_path_data_set,
                     output_file_path_processor_memory=ARGS.output_file_path_processor_memory,
                     output_file_path_target=ARGS.output_file_path_target,
                     output_file_path_predictors=ARGS.output_file_path_predictors,
                     output_file_path_engineered_feature_names=ARGS.output_file_path_engineered_feature_names,
                     re_engineering=ARGS.re_engineering,
                     next_level=ARGS.next_level,
                     feature_engineering_config=ARGS.feature_engineering_config,
                     feature_names=ARGS.feature_names,
                     exclude=ARGS.exclude,
                     exclude_original_data=ARGS.exclude_original_data,
                     sep=ARGS.sep
                     )
