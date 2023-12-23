"""

Task: ... (Function to run in container)

"""

import argparse
import pandas as pd

from aws import save_file_to_s3
from custom_logger import Log
from file_handler import file_handler
from model_generator_clustering import ModelGeneratorCluster, UnsupervisedML
from typing import Any, List, NamedTuple

MODEL_ARTIFACT_FILE_TYPE: List[str] = ['p', 'pkl', 'pickle']

PARSER = argparse.ArgumentParser(description="generate clustering models")
PARSER.add_argument('-model_name', type=str, required=True, default=None, help='name of the machine learning model')
PARSER.add_argument('-train_data_set_path', type=str, required=True, default=None, help='complete file path of the training data set')
PARSER.add_argument('-model_id', type=int, required=False, default=None, help='pre-defined model id')
PARSER.add_argument('-model_param', type=Any, required=False, default=None, help='pre-defined model hyperparameter')
PARSER.add_argument('-param_rate', type=float, required=False, default=0.0, help='')
PARSER.add_argument('-force_param', type=Any, required=False, default=None, help='immutable model hyperparameter')
PARSER.add_argument('-warm_start', type=int, required=False, default=1, help='')
PARSER.add_argument('-max_retries', type=int, required=False, default=100, help='maximum number of retries if model hyperparameter configuration raises an error')
PARSER.add_argument('-train_model', type=int, required=True, default=1, help='whether to train machine learning model or not')
PARSER.add_argument('-sep', type=str, required=False, default=',', help='file separator')
PARSER.add_argument('-prediction_variable_name', type=str, required=False, default='prediction', help='name of the prediction variable used for evaluation')
PARSER.add_argument('-kwargs', type=Any, required=False, default=None, help='pre-defined key-value arguments for process handling')
PARSER.add_argument('-output_path_model', type=str, required=True, default=None, help='file path of the model output')
PARSER.add_argument('-output_path_metadata', type=str, required=True, default=None, help='file path of the metadata output')
PARSER.add_argument('-output_path_evaluation_train_data', type=str, required=True, default=None, help='complete file path of the evaluation training data output')
PARSER.add_argument('-output_path_evaluation_data', type=str, required=True, default=None, help='file path of the evaluation data paths output')
PARSER.add_argument('-output_path_model_customized', type=str, required=False, default=None, help='file path of the evaluation data paths output')
ARGS = PARSER.parse_args()


class ModelGeneratorException(Exception):
    """
    Class for handling exceptions for function generate_model
    """
    pass


def generate_model(model_name: str,
                   train_data_set_path: str,
                   output_path_model: str,
                   output_path_metadata: str,
                   output_path_evaluation_train_data: str,
                   output_path_evaluation_data: str,
                   model_id: int = None,
                   model_param: dict = None,
                   param_rate: float = 0.0,
                   force_param: dict = None,
                   warm_start: bool = True,
                   max_retries: int = 100,
                   train_model: bool = True,
                   sep: str = ',',
                   prediction_variable_name: str = 'prediction',
                   output_path_model_customized: str = None,
                   **kwargs
                   ) -> NamedTuple('outputs', [('model_artifact', str),
                                               ('metadata', dict),
                                               ('evaluation_data', dict),
                                               ('training_status', str)
                                               ]):
    """
    Generate clustering model

    :param model_name: str
        Abbreviated name of the supervised machine learning model

    :param train_data_set_path: str
        Complete file path of the training data set

    :param output_path_model: str
        Path of the model output

    :param output_path_metadata: str
        Path of the output metadata

    :param output_path_evaluation_train_data: str
        Path of the evaluation training data set output

    :param output_path_evaluation_data: str
        Path of the evaluation data set output

    :param model_id: int
        Model ID

    :param model_param: dict
        Model hyperparameter set

    :param param_rate: float
        Rate for changing given hyperparameter set

    :param force_param: dict
        Immutable model hyperparameter set

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

    :param output_path_model_customized: str
        Complete file path of the trained model artifact

    :param kwargs: dict
        Key-word arguments for handling low and high boundaries for randomly drawing model hyperparameter configuration

    :return: NamedTuple
        Path of the sampled data sets and metadata about each data set
    """
    _file_type: str = output_path_model.split('.')[-1]
    if _file_type not in MODEL_ARTIFACT_FILE_TYPE:
        raise ModelGeneratorException(f'Model artifact file type ({_file_type}) not supported. Supported types are: {MODEL_ARTIFACT_FILE_TYPE}')
    if output_path_model_customized is not None:
        _file_type: str = output_path_model_customized.split('.')[-1]
        if _file_type not in MODEL_ARTIFACT_FILE_TYPE:
            raise ModelGeneratorException(f'Model artifact file type ({_file_type}) not supported. Supported types are: {MODEL_ARTIFACT_FILE_TYPE}')
    _model_param: dict = model_param
    if warm_start:
        _model_param = ModelGeneratorCluster(model_name=model_name, model_id=model_id, **kwargs).get_model_parameter()
    _model_generator: ModelGeneratorCluster = ModelGeneratorCluster(model_name=model_name,
                                                                    cluster_params=_model_param,
                                                                    model_id=model_id
                                                                    )
    _model_generator.generate_model()
    if 0 < param_rate < 1 and not warm_start:
        _model_generator.generate_params(param_rate=param_rate, force_param=force_param)
    _metadata: dict = dict(id=_model_generator.id,
                           param=_model_generator.model_param,
                           param_changed=_model_generator.model_param_mutated,
                           train_time_in_sec=_model_generator.train_time,
                           creation_time=_model_generator.creation_time
                           )
    if train_model:
        _train_df: pd.DataFrame = pd.read_csv(filepath_or_buffer=train_data_set_path, sep=sep)
        _features: List[str] = _train_df.columns.tolist()
        _retries: int = 0
        while _retries <= max_retries:
            try:
                _model_generator.train(x=_train_df[_features])
                _training_status: str = 'successful'
            except Exception as e:
                _retries += 1
                if _retries > round(max_retries * 0.75):
                    _model_generator.generate_model()
                else:
                    if 0 < param_rate < 1:
                        _model_generator.generate_params(param_rate=param_rate, force_param=force_param)
                    else:
                        _model_generator.generate_model()
                if _retries > max_retries:
                    _training_status: str = 'failure'
                Log().log(msg=f'Retry {_retries}: {e}')
        _metadata.update({'train_time_in_sec': _model_generator.train_time,
                          'creation_time': _model_generator.creation_time
                          })
        _pred_train, _ = _model_generator.predict(x=_train_df[_features])
        _train_df[prediction_variable_name] = _pred_train.tolist()
        _train_df.to_csv(path_or_buf=output_path_evaluation_train_data, sep=sep, header=True, index=False)
        _evaluation_data: dict = dict(train=output_path_evaluation_train_data)
        file_handler(file_path=output_path_model, obj=_model_generator.model)
        if output_path_model_customized is not None:
            save_file_to_s3(file_path=output_path_model_customized, obj=_model_generator.model)
    else:
        _evaluation_data: dict = None
        file_handler(file_path=output_path_model, obj=_model_generator)
        if output_path_model_customized is not None:
            save_file_to_s3(file_path=output_path_model_customized, obj=_model_generator)
    for file_path, obj in [(output_path_metadata, _metadata),
                           (output_path_evaluation_data, _evaluation_data),
                           (output_path_evaluation_data, _training_status)
                           ]:
        file_handler(file_path=file_path, obj=obj)
    return [_model_generator.model,
            _metadata,
            _evaluation_data,
            _training_status
            ]


if __name__ == '__main__':
    generate_model(model_name=ARGS.model_name,
                   train_data_set_path=ARGS.train_data_set_path,
                   output_path_model=ARGS.output_path_model,
                   output_path_metadata=ARGS.output_path_metadata,
                   output_path_evaluation_train_data=ARGS.output_path_evaluation_train_data,
                   output_path_evaluation_data=ARGS.output_path_evaluation_data,
                   model_id=ARGS.model_id,
                   model_param=ARGS.model_param,
                   param_rate=ARGS.param_rate,
                   force_param=ARGS.force_param,
                   warm_start=bool(ARGS.warm_start),
                   max_retries=ARGS.max_retries,
                   train_model=bool(ARGS.train_model),
                   sep=ARGS.sep,
                   prediction_variable_name=ARGS.prediction_variable_name,
                   output_path_model_customized=ARGS.output_path_model_customized,
                   **ARGS.kwargs
                   )
