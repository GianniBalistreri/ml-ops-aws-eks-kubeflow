"""

Task: ... (Function to run in container)

"""

import argparse
import ast
import pandas as pd

from aws import load_file_from_s3_as_df, save_file_to_s3, save_file_to_s3_as_df
from custom_logger import Log
from file_handler import file_handler
from sampler import FileSampler, MLSampler, Sampler
from typing import Dict, List, NamedTuple

SAMPLING_METH: List[str] = ['quota', 'random']
ML_CLF_SAMPLING_METH: List[str] = ['down', 'up']
ML_SAMPLING_METH: List[str] = ['train_test', 'train_test_time_series']

PARSER = argparse.ArgumentParser(description="data sampling")
PARSER.add_argument('-action', type=str, required=True, default=None, help='sampling action')
PARSER.add_argument('-data_set_file_path', type=str, required=True, default=None, help='complete file path to the data set')
PARSER.add_argument('-target_feature', type=str, required=True, default=None, help='name of the target feature')
PARSER.add_argument('-features', nargs='+', required=False, default=None, help='names of the features')
PARSER.add_argument('-time_series_feature', type=str, required=False, default=None, help='name of the time series feature')
PARSER.add_argument('-train_size', type=float, required=False, default=0.8, help='size of the training data set')
PARSER.add_argument('-validation_size', type=float, required=False, default=0.1, help='size of the validation data set')
PARSER.add_argument('-random_sample', type=int, required=False, default=1, help='whether to sample randomly or not')
PARSER.add_argument('-target_class_value', type=int, required=False, default=None, help='target class value to sample')
PARSER.add_argument('-target_proportion', type=float, required=False, default=None, help='target proportion of class value')
PARSER.add_argument('-size', type=int, required=False, default=None, help='size of the sampled data set')
PARSER.add_argument('-prop', type=float, required=False, default=None, help='proportion of the sampled data set')
PARSER.add_argument('-quotas', type=str, required=False, default=None, help='pre-defined quota configuration for sampling')
PARSER.add_argument('-sep', type=str, required=False, default=',', help='column separator')
PARSER.add_argument('-output_file_path_sampling_metadata', type=str, required=True, default=None, help='complete file path of the metadata output')
PARSER.add_argument('-output_file_path_sampling_file_paths', type=str, required=True, default=None, help='complete file path of the file sampling output')
PARSER.add_argument('-s3_output_file_path_train_data_set', type=str, required=False, default=None, help='complete file path of the training data set output')
PARSER.add_argument('-s3_output_file_path_test_data_set', type=str, required=False, default=None, help='complete file path of the test data set output')
PARSER.add_argument('-s3_output_file_path_val_data_set', type=str, required=False, default=None, help='complete file path of the validation data set output')
PARSER.add_argument('-s3_output_file_path_sampling_data_set', type=str, required=False, default=None, help='S3 file path of the sampled data set output')
PARSER.add_argument('-s3_output_file_path_sampling_metadata', type=str, required=False, default=None, help='S3 file path of the sampling metadata output')
ARGS = PARSER.parse_args()


class SamplingException(Exception):
    """
    Class for handling exceptions for function sampling
    """
    pass


def sampling(action: str,
             data_set_file_path: str,
             target_feature: str,
             output_file_path_sampling_metadata: str = None,
             output_file_path_sampling_file_paths: str = None,
             s3_output_file_path_train_data_set: str = None,
             s3_output_file_path_test_data_set: str = None,
             s3_output_file_path_val_data_set: str = None,
             s3_output_file_path_sampling_data_set: str = None,
             features: List[str] = None,
             time_series_feature: str = None,
             train_size: float = 0.8,
             validation_size: float = 0.1,
             random_sample: bool = True,
             target_class_value: int = None,
             target_proportion: float = None,
             size: int = None,
             prop: float = None,
             quotas: Dict[str, Dict[str, float]] = None,
             sep: str = ',',
             s3_output_file_path_sampling_metadata: str = None,
             ) -> NamedTuple('outputs', [('sampling_metadata', dict),
                                         ('sampling_file_paths', list)
                                         ]
                             ):
    """
    Sampling data sets for training, testing and validation used for applying supervised machine learning models

    :param action: str
        Name of the sampling action
            -> random: Random sampling
            -> quota: Quota based sampling
            -> down: Down-sampling of class value
            -> up: Up-sampling of class value
            -> train_test: Train-test sampling for structured data
            -> train_test_time_series: Train-test sampling for time series data
            -> file_random: Random file sampling
            -> file_first: First n files sampling
            -> file_last: Last n files sampling

    :param data_set_file_path: str
        Complete file path of the data set

    :param target_feature: str
        Name of the target feature

    :param output_file_path_sampling_metadata: str
        File path of the sampling metadata output

    :param output_file_path_sampling_file_paths: str
        File path of the file sampling output

    :param s3_output_file_path_train_data_set: str
        Complete file path of the sampled training data set

    :param s3_output_file_path_test_data_set: str
        Complete file path of the sampled test data set

    :param s3_output_file_path_val_data_set: str
        Complete file path of the sampled validation data set

    :param s3_output_file_path_sampling_data_set: str
        Complete file path of the sampled data set

    :param features: List[str]
        Name of features to use

    :param time_series_feature: str
        Name of the datetime feature to use

    :param train_size: float
        Size of the training data set

    :param validation_size: float
        Size of the validation data set

    :param random_sample: bool
        Whether to sample randomly or not

    :param target_class_value: Union[str, int]
        Class value of the target feature to sample

    :param target_proportion: float
        Target proportion of the class value of the target feature

    :param size: int
        Sample size

    :param prop: float
        Proportion of the sample size

    :param quotas: Dict[str, Dict[str, float]]
        Pre-defined quota config used for quota sampling

    :param sep: str
        Separator

    :param s3_output_file_path_sampling_metadata: str
        Complete file path of the sampling metadata

    :return: NamedTuple
        Metadata about sampled data sets and file paths of the sampled files
    """
    _cpu_available: int = get_available_cpu(logging=True)
    _memory_total: float = get_memory(total=True, logging=True)
    _memory_available: float = get_memory(total=False, logging=True)
    _df: pd.DataFrame = load_file_from_s3_as_df(file_path=data_set_file_path, sep=sep)
    Log().log(msg=f'Load data set: {data_set_file_path} -> Cases={_df.shape[0]}, Features={_df.shape[1]}')
    _file_paths: List[str] = []
    if action in SAMPLING_METH or action in ML_CLF_SAMPLING_METH:
        if action in SAMPLING_METH:
            _sampler: Sampler = Sampler(df=_df, size=size, prop=prop)
        else:
            _sampler: MLSampler = MLSampler(df=_df,
                                            target=target_feature,
                                            features=features,
                                            time_series_feature=time_series_feature,
                                            train_size=train_size,
                                            validation_size=validation_size,
                                            random_sample=random_sample,
                                            stratification=False
                                            )
        if action == 'quota':
            _sampled_df: pd.DataFrame = _sampler.quota(features=features, quotas=quotas)
        elif action == 'random':
            _sampled_df: pd.DataFrame = _sampler.random()
        elif action == 'down':
            _sampled_df = _sampler.down_sampling(target_class_value=target_class_value, target_proportion=target_proportion)
        else:
            _sampled_df = _sampler.up_sampling(target_class_value=target_class_value, target_proportion=target_proportion)
        _sampling_metadata: dict = dict(n_features=_sampled_df.shape[1], n_cases={action: _sampled_df.shape[0]})
        save_file_to_s3_as_df(file_path=s3_output_file_path_sampling_data_set, df=_sampled_df, sep=sep)
        Log().log(msg=f'Save {action} sampled data set: {s3_output_file_path_sampling_data_set} -> Cases={_sampled_df.shape[0]}, Features={_sampled_df.shape[1]}')
    elif action in ML_SAMPLING_METH:
        _ml_sampler: MLSampler = MLSampler(df=_df,
                                           target=target_feature,
                                           features=features,
                                           time_series_feature=time_series_feature,
                                           train_size=train_size,
                                           validation_size=validation_size,
                                           random_sample=random_sample,
                                           stratification=False
                                           )
        if action == 'train_test':
            _train_test_split: dict = _ml_sampler.train_test_sampling()
        else:
            _train_test_split: dict = _ml_sampler.time_series_sampling()
        _train_df: pd.DataFrame = _train_test_split.get('x_train')
        _train_df[target_feature] = _train_test_split.get('y_train')
        save_file_to_s3_as_df(file_path=s3_output_file_path_train_data_set, df=_train_df, sep=sep)
        Log().log(msg=f'Save training data set: {s3_output_file_path_train_data_set}')
        _test_df: pd.DataFrame = _train_test_split.get('x_test')
        _test_df[target_feature] = _train_test_split.get('y_test')
        save_file_to_s3_as_df(file_path=s3_output_file_path_test_data_set, df=_test_df, sep=sep)
        Log().log(msg=f'Save test data set: {s3_output_file_path_test_data_set}')
        _sampling_metadata: dict = dict(n_features=_train_df.shape[1] - 1,
                                        n_cases={'train': _train_df.shape[0], 'test': _test_df.shape[0]}
                                        )
        if _train_test_split.get('x_val') is not None and _train_test_split.get('y_val') is not None:
            _val_df: pd.DataFrame = _train_test_split.get('x_val')
            _val_df[target_feature] = _train_test_split.get('y_val')
            save_file_to_s3_as_df(file_path=s3_output_file_path_val_data_set, df=_val_df, sep=sep)
            Log().log(msg=f'Save validation data set: {s3_output_file_path_val_data_set}')
            _sampling_metadata['n_cases'].update({'val': _val_df.shape[0]})
    elif action.find('file') >= 0:
        _sampler: FileSampler = FileSampler(file_path=data_set_file_path, size=size, prop=prop)
        _file_paths: List[str] = _sampler.main(meth=action.split('file_')[-1])
        _sampling_metadata: dict = dict(n_files=len(_file_paths))
    else:
        raise SamplingException(f'Sampling action ({action}) not supported')
    file_handler(file_path=output_file_path_sampling_metadata, obj=_sampling_metadata)
    file_handler(file_path=output_file_path_sampling_file_paths, obj=_file_paths)
    if s3_output_file_path_sampling_metadata is not None:
        save_file_to_s3(file_path=s3_output_file_path_sampling_metadata, obj=_sampling_metadata)
    _cpu_utilization: float = get_cpu_utilization(interval=1, logging=True)
    _cpu_utilization_per_cpu: List[float] = get_cpu_utilization_per_core(interval=1, logging=True)
    _memory_utilization: float = get_memory_utilization(logging=True)
    _memory_available = get_memory(total=False, logging=True)
    return [_sampling_metadata, _file_paths]


if __name__ == '__main__':
    if ARGS.features:
        ARGS.features = ast.literal_eval(ARGS.features[0])
    if ARGS.quotas:
        ARGS.quotas = ast.literal_eval(ARGS.quotas)
    sampling(action=ARGS.action,
             data_set_file_path=ARGS.data_set_file_path,
             target_feature=ARGS.target_feature,
             output_file_path_sampling_metadata=ARGS.output_file_path_sampling_metadata,
             output_file_path_sampling_file_paths=ARGS.output_file_path_sampling_file_paths,
             s3_output_file_path_train_data_set=ARGS.s3_output_file_path_train_data_set,
             s3_output_file_path_test_data_set=ARGS.s3_output_file_path_test_data_set,
             s3_output_file_path_val_data_set=ARGS.s3_output_file_path_val_data_set,
             s3_output_file_path_sampling_data_set=ARGS.s3_output_file_path_sampling_data_set,
             features=ARGS.features,
             time_series_feature=ARGS.time_series_feature,
             train_size=ARGS.train_size,
             validation_size=ARGS.validation_size,
             random_sample=bool(ARGS.random_sample),
             target_class_value=ARGS.target_class_value,
             target_proportion=ARGS.target_proportion,
             size=ARGS.size,
             prop=ARGS.prop,
             quotas=ARGS.quotas,
             sep=ARGS.sep,
             s3_output_file_path_sampling_metadata=ARGS.s3_output_file_path_sampling_metadata
             )
