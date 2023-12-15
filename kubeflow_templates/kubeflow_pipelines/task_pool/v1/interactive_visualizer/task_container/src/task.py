"""

Task: ... (Function to run in container)

"""

import argparse
import boto3
import json
import os
import pandas as pd
import pickle

from custom_logger import Log
from interactive_visualizer import ENGINEERING_METH, MIN_FEATURES_BY_METH, FeatureEngineer
from typing import Any, Dict, NamedTuple, List

PARSER = argparse.ArgumentParser(description="feature importance")
PARSER.add_argument('-data_set_path', type=str, required=True, default=None, help='file path of the data set')
PARSER.add_argument('-analytical_data_types', type=Any, required=True, default=None, help='assignment of features to analytical data types')
PARSER.add_argument('-target_feature_name', type=str, required=True, default=None, help='name of the target feature')
PARSER.add_argument('-output_bucket_name', type=str, required=True, default=None, help='name of the S3 output bucket')
PARSER.add_argument('-output_file_path_data_set', type=str, required=True, default=None, help='file path of the data set')
PARSER.add_argument('-output_file_path_processor_obj', type=str, required=True, default=None, help='file path of output processor objects')
PARSER.add_argument('-output_file_path_target', type=str, required=True, default=None, help='file path of the output target feature')
PARSER.add_argument('-output_file_path_predictors', type=str, required=True, default=None, help='file path of the output predictors')
PARSER.add_argument('-output_file_path_engineered_feature_names', type=str, required=True, default=None, help='file path of the output processed features')
PARSER.add_argument('-feature_engineering_config', type=Any, required=False, default=None, help='feature engineering pre-defined config file')
PARSER.add_argument('-feature_names', type=str, required=False, default=None, help='pre-defined feature names used for feature engineering')
PARSER.add_argument('-exclude', type=str, required=False, default=None, help='pre-defined feature names to exclude')
PARSER.add_argument('-exclude_original_data', type=str, required=False, default=None, help='exclude all original features (especially numeric features)')
PARSER.add_argument('-sep', type=str, required=False, default=',', help='column separator')
ARGS = PARSER.parse_args()


def feature_importance(data_set_path: str,
                     analytical_data_types: dict,
                     target_feature_name: str,
                     output_bucket_name: str,
                     output_file_path_data_set: str,
                     output_file_path_processor_obj: str,
                     output_file_path_target: str,
                     output_file_path_predictors: str,
                     output_file_path_engineered_feature_names: str,
                     feature_engineering_config: Dict[str, list] = None,
                     feature_names: List[str] = None,
                     exclude: List[str] = None,
                     exclude_original_data: bool = False,
                     sep: str = ','
                     ) -> NamedTuple(typename='outputs', fields=[('file_path_data', str),
                                                                 ('file_path_processor_obj', str),
                                                                 ('engineered_feature_names', list),
                                                                 ('predictors', list),
                                                                 ('target', str)
                                                                 ]
                                     ):
    """
    Feature importance of structured (tabular) data

    :param data_set_path: str
        Complete file path of the data set

    :param analytical_data_types: dict
        Assigned analytical data types to each feature

    :param output_bucket_name: str
        Name of the output S3 bucket

    :param output_file_path_processor_obj: str
        Path of the processor objects to save

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
                            if feature in _features:
                                _feature_engineering_config[meth].append(feature)
    else:
        _feature_engineering_config: Dict[str, list] = feature_engineering_config
    _feature_engineer: FeatureEngineer = FeatureEngineer(df=_df)
    _df_engineered, _processor_objs, _non_numeric_features = _feature_engineer.main(feature_engineering_config=_feature_engineering_config)
    pd.concat(objs=[_df, _df_engineered], axis=1).to_csv(path_or_buf=os.path.join(output_bucket_name, output_file_path_data_set), sep=sep, index=False)
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
    for non_numeric in _non_numeric_features:
        if non_numeric in _predictors:
            del _predictors[_predictors.index(non_numeric)]
            Log().log(msg=f'Exclude original (non-numeric) feature "{non_numeric}"')
    _predictors = sorted(_predictors)
    for file_path, obj in [(output_file_path_data_set, output_file_path_data_set),
                           (output_file_path_target, target_feature_name),
                           (output_file_path_predictors, _predictors),
                           (output_file_path_engineered_feature_names, _feature_names_engineered)
                           ]:
        with open(file_path) as _file:
            json.dump(obj, _file)
    _s3_resource: boto3 = boto3.resource('s3')
    _s3_model_obj: _s3_resource.Object = _s3_resource.Object(output_bucket_name, output_file_path_processor_obj)
    _s3_model_obj.put(Body=pickle.dumps(obj=_processor_objs, protocol=pickle.HIGHEST_PROTOCOL))
    return [output_file_path_data_set,
            output_file_path_processor_obj,
            _feature_names_engineered,
            _predictors,
            target_feature_name
            ]


if __name__ == '__main__':
    feature_importance(data_set_path=ARGS.data_set_path,
                     analytical_data_types=ARGS.analytical_data_types,
                     target_feature_name=ARGS.target_feature_name,
                     output_bucket_name=ARGS.output_bucket_name,
                     output_file_path_data_set=ARGS.output_file_path_data_set,
                     output_file_path_processor_obj=ARGS.output_file_path_processor_obj,
                     output_file_path_target=ARGS.output_file_path_target,
                     output_file_path_predictors=ARGS.output_file_path_predictors,
                     output_file_path_engineered_feature_names=ARGS.output_file_path_engineered_feature_names,
                     feature_engineering_config=ARGS.feature_engineering_config,
                     feature_names=ARGS.feature_names,
                     exclude=ARGS.exclude,
                     exclude_original_data=ARGS.exclude_original_data,
                     sep=ARGS.sep
                     )
