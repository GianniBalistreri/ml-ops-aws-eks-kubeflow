"""

Task: ... (Function to run in container)

"""

import argparse
import boto3
import json
import os
import numpy as np
import pandas as pd
import pickle

from datetime import datetime
from supervised_machine_learning import ModelGeneratorClf, ModelGeneratorReg
from typing import NamedTuple, List


PARSER = argparse.ArgumentParser(description="generate supervised machine learning models")
PARSER.add_argument('-msg', type=str, required=True, default=None, help='slack message to send')
PARSER.add_argument('-pipeline_status', type=str, required=True, default=None, help='pipeline status')
PARSER.add_argument('-aws_region', type=str, required=False, default='eu-central-1', help='AWS region code')
PARSER.add_argument('-slack_channel', type=str, required=False, default='slack-arcana-analytica', help='name of the destined slack channel')
ARGS = PARSER.parse_args()


def generate_model(ml_type: str,
                   target_feature_name: str,
                   train_data_set_path: str,
                   test_data_set_path: str,
                   val_data_set_path: str,
                   output_path_model: str,
                   output_path_metadata: str,
                   model_name: str,
                   model_param: dict = None,
                   model_id: int = None,
                   mutate_or_adjust: bool = False,
                   train_model: bool = False,
                   sep: str = ',',
                   ) -> NamedTuple('outputs', [('model', str),
                                               ('metadata', str),
                                               ('param', dict),
                                               ('training_duration_in_sec', int)
                                               ]):
    """
    Generate supervised machine learning model

    :param ml_type: str
        Name of the machine learning problem
            -> reg: Regression
            -> clf_binary: Binary Classification
            -> clf_multi: Multi-Classification

    :param target_feature_name: str
        Name of the target feature

    :param train_data_set_path: str
        Complete file path of the training data set

    :param test_data_set_path: str
        Complete file path of the tests data set

    :param val_data_set_path: str
        Complete file path of the validation data set

    :param output_path_model: str
        Path of the model to save

    :param output_path_metadata: str
        Path of the metadata to save

    :param project_name: str
        Name of the project (used in file names)

    :param model_name: str
        Abbreviated name of the supervised machine learning model

    :param model_param: dict
        Model parameter configuration used for training

    :param model_id: int
        Model identifier (used in evolutionary framework)

    :param mutate_or_adjust: bool
        Whether to mutate or adjust hyperparameter of the model (used in evolutionary framework)

    :param sep: str
        Separator

    :return: NamedTuple
        Path of the sampled data sets and metadata about each data set
    """
    _s3_resource: boto3 = boto3.resource('s3')
    if ml_type == 'reg':
        _model_generator: ModelGeneratorReg = ModelGeneratorReg(model_name=model_name, reg_params=model_param)
    else:
        _model_generator: ModelGeneratorClf = ModelGeneratorClf(model_name=model_name, clf_params=model_param)
    _model_generator.generate_model()
    if mutate_or_adjust:
        _model_generator.generate_params(param_rate=0.1, force_param=None)
    if train_model:
        _train_df: pd.DataFrame = pd.read_csv(filepath_or_buffer=train_data_set_path, sep=sep)
        _test_df: pd.DataFrame = pd.read_csv(filepath_or_buffer=test_data_set_path, sep=sep)
        _val_df: pd.DataFrame = pd.read_csv(filepath_or_buffer=val_data_set_path, sep=sep)
        _features: List[str] = _train_df.columns.tolist()
        if target_feature_name in _features:
            del _features[_features.index(target_feature_name)]
        _model_generator.train(x=_train_df[_features], y=_train_df[target_feature_name])
        _pred_train: np.ndarray = _model_generator.predict(x=_train_df[_features])
        _pred_test: np.ndarray = _model_generator.predict(x=_test_df[_features])
        _pred_val: np.ndarray = _model_generator.predict(x=_val_df[_features])
        _train_df['prediction'] = _pred_train.tolist()
        _test_df['prediction'] = _pred_test.tolist()
        _val_df['prediction'] = _pred_val.tolist()
        _train_df.to_csv(path_or_buf='', sep=sep, header=True, index=False)
        _test_df.to_csv(path_or_buf='', sep=sep, header=True, index=False)
        _val_df.to_csv(path_or_buf='', sep=sep, header=True, index=False)
        if model_id is None:
            _model_file_name: str = f'{project_name}_model_{str(datetime.now())}.p'
        else:
            _model_file_name: str = f'{project_name}_model_{model_id}_{str(datetime.now())}.p'
        _s3_model_obj: _s3_resource.Object = _s3_resource.Object(output_path_model, _model_file_name)
        _s3_model_obj.put(Body=pickle.dumps(obj=_model_generator.model, protocol=pickle.HIGHEST_PROTOCOL))
    else:
        _s3_model_obj: _s3_resource.Object = _s3_resource.Object(output_path_model, _model_file_name)
        _s3_model_obj.put(Body=pickle.dumps(obj=_model_generator, protocol=pickle.HIGHEST_PROTOCOL))
    return [_model_file_name, _model_generator.model_param, _model_generator.train_time]


if __name__ == '__main__':
    generate_model(ml_type=ARGS.ml_type,
                   target_feature_name=ARGS.target_feature_name,
                   train_data_set_path=ARGS.train_data_set_path,
                   test_data_set_path=ARGS.test_data_set_path,
                   val_data_set_path=ARGS.val_data_set_path,
                   output_path_model=ARGS.output_path_model,
                   output_path_metadata=ARGS.output_path_metadata,
                   model_name=ARGS.model_name,
                   model_param=ARGS.model_param,
                   model_id=ARGS.model_id,
                   mutate_or_adjust=ARGS.mutate_or_adjust,
                   train_model=ARGS.train_model,
                   sep=ARGS.sep
                   )
