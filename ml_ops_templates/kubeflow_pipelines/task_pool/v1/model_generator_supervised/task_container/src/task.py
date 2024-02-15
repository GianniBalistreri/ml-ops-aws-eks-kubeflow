"""

Task: ... (Function to run in container)

"""

import argparse
import ast
import numpy as np
import pandas as pd

from aws import load_file_from_s3, load_file_from_s3_as_df, save_file_to_s3, save_file_to_s3_as_df
from custom_logger import Log
from file_handler import file_handler
from supervised_machine_learning import ModelGeneratorClf, ModelGeneratorReg
from typing import List, NamedTuple

MODEL_ARTIFACT_FILE_TYPE: List[str] = ['joblib', 'p', 'pkl', 'pickle']

PARSER = argparse.ArgumentParser(description="generate non-neural network supervised machine learning models")
PARSER.add_argument('-ml_type', type=str, required=True, default=None, help='name of the machine learning type')
PARSER.add_argument('-model_name', type=str, required=True, default=None, help='name of the machine learning model')
PARSER.add_argument('-target_feature', type=str, required=True, default=None, help='name of the target feature')
PARSER.add_argument('-predictors', nargs='+', required=False, default=None, help='predictor names')
PARSER.add_argument('-train_data_set_path', type=str, required=True, default=None, help='complete file path of the training data set')
PARSER.add_argument('-test_data_set_path', type=str, required=True, default=None, help='complete file path of the test data set')
PARSER.add_argument('-val_data_set_path', type=str, required=False, default=None, help='complete file path of the validation data set')
PARSER.add_argument('-model_id', type=int, required=False, default=None, help='pre-defined model id')
PARSER.add_argument('-model_param_path', type=str, required=False, default=None, help='file path of the pre-defined model hyperparameter')
PARSER.add_argument('-param_rate', type=float, required=False, default=0.0, help='rate for changing hyperparameter set')
PARSER.add_argument('-force_param_path', type=str, required=False, default=None, help='file path of the immutable model hyperparameter')
PARSER.add_argument('-warm_start', type=int, required=False, default=1, help='Whether to use standard hyperparameter set or not')
PARSER.add_argument('-max_retries', type=int, required=False, default=100, help='maximum number of retries if model hyperparameter configuration raises an error')
PARSER.add_argument('-train_model', type=int, required=True, default=1, help='whether to train machine learning model or not')
PARSER.add_argument('-sep', type=str, required=False, default=',', help='file separator')
PARSER.add_argument('-prediction_variable_name', type=str, required=False, default='prediction', help='name of the prediction variable used for evaluation')
PARSER.add_argument('-parallel_mode', type=int, required=False, default=0, help='whether to run task in parallel mode or not')
PARSER.add_argument('-kwargs', type=str, required=False, default=None, help='pre-defined key-value arguments for process handling')
PARSER.add_argument('-output_path_training_status', type=str, required=True, default=None, help='file path of the training status output')
PARSER.add_argument('-s3_output_path_evaluation_train_data', type=str, required=True, default=None, help='complete file path of the evaluation training data output')
PARSER.add_argument('-s3_output_path_evaluation_test_data', type=str, required=True, default=None, help='complete file path of the evaluation test data output')
PARSER.add_argument('-s3_output_path_evaluation_val_data', type=str, required=False, default=None, help='complete file path of the evaluation validation data output')
PARSER.add_argument('-s3_output_path_metadata', type=str, required=True, default=None, help='complete file path of the metadata output')
PARSER.add_argument('-s3_output_path_param', type=str, required=False, default=None, help='complete file path of the hyperparameter output')
PARSER.add_argument('-s3_output_path_model', type=str, required=False, default=None, help='S3 file path of the evaluation data output')
ARGS = PARSER.parse_args()


class ModelGeneratorException(Exception):
    """
    Class for handling exceptions for function generate_model
    """
    pass


def generate_model(ml_type: str,
                   model_name: str,
                   target_feature: str,
                   train_data_set_path: str,
                   test_data_set_path: str,
                   s3_output_path_model: str,
                   s3_output_path_param: str,
                   s3_output_path_metadata: str,
                   s3_output_path_evaluation_train_data: str,
                   s3_output_path_evaluation_test_data: str,
                   output_path_training_status: str,
                   predictors: List[str] = None,
                   model_id: int = None,
                   model_param_path: str = None,
                   param_rate: float = 0.0,
                   force_param_path: str = None,
                   warm_start: bool = True,
                   max_retries: int = 100,
                   train_model: bool = True,
                   sep: str = ',',
                   prediction_variable_name: str = 'prediction',
                   parallel_mode: bool = False,
                   val_data_set_path: str = None,
                   s3_output_path_evaluation_val_data: str = None,
                   **kwargs
                   ) -> NamedTuple('outputs', [('training_status', str)]):
    """
    Generate supervised machine learning model

    :param ml_type: str
        Name of the machine learning problem
            -> reg: Regression
            -> clf_binary: Binary Classification
            -> clf_multi: Multi-Classification

    :param model_name: str
        Abbreviated name of the supervised machine learning model

    :param target_feature: str
        Name of the target feature

    :param train_data_set_path: str
        Complete file path of the training data set

    :param test_data_set_path: str
        Complete file path of the tests data set

    :param s3_output_path_model: str
        Complete S3 file path of the trained model artifact

    :param s3_output_path_param: str
        Complete file path of the hyperparameter output

    :param s3_output_path_metadata: str
        Complete file path of the metadata output

    :param s3_output_path_evaluation_train_data: str
        Complete file path of the evaluation training data set output

    :param s3_output_path_evaluation_test_data: str
        Complete file path of the evaluation test data set output

    :param output_path_training_status: str
        Path of the training status output

    :param predictors: List[str]
        Name of the predictors

    :param model_id: int
        Model ID

    :param model_param_path: str
        Complete file path of the model hyperparameter set

    :param param_rate: float
        Rate for changing given hyperparameter set

    :param force_param_path: str
        Complete file path of the immutable model hyperparameter set

    :param warm_start: bool
        Whether to use standard hyperparameter set or not

    :param max_retries: int
        Maximum number of retries if model hyperparameter configuration raises an error

    :param train_model: bool
        Whether to train model or not

    :param sep: str
        Separator

    :param prediction_variable_name: str
        Name of the prediction variable for evaluation step afterward

    :param parallel_mode: bool
        Whether to run task in parallel mode or not

    :param val_data_set_path: str
        Complete file path of the validation data set

    :param s3_output_path_evaluation_val_data: str
        Complete file path of the evaluation validation data set output

    :param kwargs: dict
        Key-word arguments for handling low and high boundaries for randomly drawing model hyperparameter configuration

    :return: NamedTuple
        Training status
    """
    _file_type: str = s3_output_path_model.split('.')[-1]
    if _file_type not in MODEL_ARTIFACT_FILE_TYPE:
        raise ModelGeneratorException(f'Model artifact file type ({_file_type}) not supported. Supported types are: {MODEL_ARTIFACT_FILE_TYPE}')
    _training_status: str = 'unknown'
    if model_param_path is None:
        _model_param: dict = None
    else:
        if len(model_param_path) == 0:
            _model_param: dict = None
        else:
            _model_param: dict = load_file_from_s3(file_path=model_param_path)
    if force_param_path is None:
        _force_param: dict = None
    else:
        if len(force_param_path) == 0:
            _force_param: dict = None
        else:
            _force_param: dict = load_file_from_s3(file_path=force_param_path)
    if ml_type == 'reg':
        if warm_start:
            _model_param = ModelGeneratorReg(model_name=model_name, model_id=model_id, **kwargs).get_standard_model_parameter()
        _model_generator: ModelGeneratorReg = ModelGeneratorReg(model_name=model_name,
                                                                reg_params=_model_param,
                                                                model_id=model_id,
                                                                **kwargs
                                                                )
    else:
        if warm_start:
            _model_param = ModelGeneratorClf(model_name=model_name, model_id=model_id, **kwargs).get_standard_model_parameter()
        _model_generator: ModelGeneratorClf = ModelGeneratorClf(model_name=model_name,
                                                                clf_params=_model_param,
                                                                model_id=model_id,
                                                                **kwargs
                                                                )
    _model_generator.generate_model()
    if 0 < param_rate < 1 and not warm_start:
        _model_generator.generate_params(param_rate=param_rate, force_param=_force_param)
    _metadata: dict = dict(id=_model_generator.id,
                           param_changed=_model_generator.model_param_mutated,
                           train_time_in_sec=_model_generator.train_time,
                           creation_time=_model_generator.creation_time
                           )
    _s3_output_path_model: str = s3_output_path_model
    if train_model:
        _train_df: pd.DataFrame = load_file_from_s3_as_df(file_path=train_data_set_path, sep=sep)
        Log().log(msg=f'Load training data set: {train_data_set_path}')
        _test_df: pd.DataFrame = load_file_from_s3_as_df(file_path=test_data_set_path, sep=sep)
        Log().log(msg=f'Load test data set: {test_data_set_path}')
        _features: List[str] = _train_df.columns.tolist() if predictors is None else predictors
        if target_feature in _features:
            del _features[_features.index(target_feature)]
        _retries: int = 0
        while _retries <= max_retries:
            try:
                _model_generator.train(x=_train_df[_features], y=_train_df[target_feature])
                _training_status: str = 'successful'
                break
            except Exception as e:
                _retries += 1
                if _retries > round(max_retries * 0.75):
                    _model_generator.generate_model()
                else:
                    if 0 < param_rate < 1:
                        _model_generator.generate_params(param_rate=param_rate, force_param=_force_param)
                    else:
                        _model_generator.generate_model()
                if _retries > max_retries:
                    _training_status: str = 'failure'
                Log().log(msg=f'Retry {_retries}: {e}')
        Log().log(msg=f'Training status: {_training_status}')
        _metadata.update({'train_time_in_sec': _model_generator.train_time,
                          'creation_time': _model_generator.creation_time
                          })
        _pred_train: np.ndarray = _model_generator.predict(x=_train_df[_features])
        _pred_test: np.ndarray = _model_generator.predict(x=_test_df[_features])
        _train_df[prediction_variable_name] = _pred_train.tolist()
        _test_df[prediction_variable_name] = _pred_test.tolist()
        _s3_output_path_evaluation_train_data: str = s3_output_path_evaluation_train_data
        if parallel_mode:
            _s3_output_path_evaluation_train_data = s3_output_path_evaluation_train_data.replace('.', f'_{_model_generator.id}.')
        save_file_to_s3_as_df(file_path=_s3_output_path_evaluation_train_data, df=_train_df, sep=sep)
        Log().log(msg=f'Save training data set for evaluation: {_s3_output_path_evaluation_train_data}')
        _s3_output_path_evaluation_test_data: str = s3_output_path_evaluation_test_data
        if parallel_mode:
            _s3_output_path_evaluation_test_data = s3_output_path_evaluation_test_data.replace('.', f'_{_model_generator.id}.')
        save_file_to_s3_as_df(file_path=_s3_output_path_evaluation_test_data, df=_test_df, sep=sep)
        Log().log(msg=f'Save test data set for evaluation: {_s3_output_path_evaluation_test_data}')
        _s3_output_path_evaluation_val_data: str = s3_output_path_evaluation_val_data
        if s3_output_path_evaluation_val_data is not None:
            _val_df: pd.DataFrame = load_file_from_s3_as_df(file_path=val_data_set_path, sep=sep)
            _pred_val: np.ndarray = _model_generator.predict(x=_val_df[_features])
            _val_df[prediction_variable_name] = _pred_val.tolist()
            _s3_output_path_evaluation_val_data: str = _s3_output_path_evaluation_val_data
            if parallel_mode:
                _s3_output_path_evaluation_val_data = _s3_output_path_evaluation_val_data.replace('.', f'_{_model_generator.id}.')
            save_file_to_s3_as_df(file_path=_s3_output_path_evaluation_val_data, df=_val_df, sep=sep)
            Log().log(msg=f'Save validation data set for evaluation: {_s3_output_path_evaluation_val_data}')
        _s3_output_path_param: str = s3_output_path_param
        if parallel_mode:
            _s3_output_path_model = s3_output_path_model.replace('.', f'_{_model_generator.id}.')
            _s3_output_path_param = s3_output_path_param.replace('.', f'_{_model_generator.id}.')
        save_file_to_s3(file_path=_s3_output_path_model, obj=_model_generator.model)
        Log().log(msg=f'Save trained model artifact: {_s3_output_path_model}')
        save_file_to_s3(file_path=_s3_output_path_param, obj=_model_generator.model_param)
        Log().log(msg=f'Save trained model param: {_s3_output_path_param}')
        _metadata.update({'param': _s3_output_path_param})
    else:
        if parallel_mode:
            _s3_output_path_model = s3_output_path_model.replace('.', f'_{_model_generator.id}.')
        save_file_to_s3(file_path=_s3_output_path_model, obj=_model_generator)
        Log().log(msg=f'Save model generator artifact: {_s3_output_path_model}')
    _s3_output_path_metadata: str = s3_output_path_metadata
    if parallel_mode:
        _s3_output_path_metadata = _s3_output_path_metadata.replace('.', f'_{_model_generator.id}.')
    save_file_to_s3(file_path=_s3_output_path_metadata, obj=_metadata)
    Log().log(msg=f'Save metadata: {_s3_output_path_metadata}')
    file_handler(file_path=output_path_training_status, obj=_training_status)
    return [_training_status]


if __name__ == '__main__':
    if ARGS.predictors:
        ARGS.predictors = ast.literal_eval(ARGS.predictors[0])
    if ARGS.kwargs:
        ARGS.kwargs = ast.literal_eval(ARGS.kwargs)
    generate_model(ml_type=ARGS.ml_type,
                   model_name=ARGS.model_name,
                   target_feature=ARGS.target_feature,
                   predictors=ARGS.predictors,
                   train_data_set_path=ARGS.train_data_set_path,
                   test_data_set_path=ARGS.test_data_set_path,
                   s3_output_path_metadata=ARGS.s3_output_path_metadata,
                   s3_output_path_param=ARGS.s3_output_path_param,
                   s3_output_path_evaluation_train_data=ARGS.s3_output_path_evaluation_train_data,
                   s3_output_path_evaluation_test_data=ARGS.s3_output_path_evaluation_test_data,
                   output_path_training_status=ARGS.output_path_training_status,
                   model_id=ARGS.model_id,
                   model_param_path=ARGS.model_param_path,
                   param_rate=ARGS.param_rate,
                   force_param_path=ARGS.force_param_path,
                   warm_start=bool(ARGS.warm_start),
                   max_retries=ARGS.max_retries,
                   train_model=bool(ARGS.train_model),
                   sep=ARGS.sep,
                   prediction_variable_name=ARGS.prediction_variable_name,
                   parallel_mode=bool(ARGS.parallel_mode),
                   val_data_set_path=ARGS.val_data_set_path,
                   s3_output_path_evaluation_val_data=ARGS.s3_output_path_evaluation_val_data,
                   s3_output_path_model=ARGS.s3_output_path_model,
                   **ARGS.kwargs
                   )
