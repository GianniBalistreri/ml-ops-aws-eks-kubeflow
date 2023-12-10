"""
Task: ... (Function to run in container)
"""

import boto3
import json
import os
import pandas as pd

from sampler import MLSampler
from typing import NamedTuple, List


def train_test_split(data_set_file_path: str,
                     target_feature_name: str,
                     output_file_path_train_test_split_data: str,
                     output_file_path_sampling_metadata: str,
                     output_bucket_name: str = None,
                     features: List[str] = None,
                     train_size: float = 0.8,
                     validation_size: float = 0.1,
                     random_sample: bool = True,
                     sep: str = ',',
                     seed: int = 1234
                     ) -> NamedTuple('outputs', [('train_data_set_path', str),
                                                 ('test_data_set_path', str),
                                                 ('val_data_set_path', str),
                                                 ('metadata', dict)
                                                 ]):
    """
    Sampling data sets for training, testing and validation used for applying supervised machine learning models

    :param data_set_file_path: str
        Complete file path of the data set

    :param output_path: str
        Path of the sample data sets

    :param target_feature_name: str
        Name of the target feature

    :param features: List[str]
        Name of features to use

    :param train_size: float
        Size of the training data set

    :param validation_size: float
        Size of the validation data set

    :param random_sample: bool
        Whether to sample randomly or not

    :param sep: str
        Separator

    :param seed: int
        Seed value

    :return: NamedTuple
        Path of the sampled data sets and metadata about each data set
    """
    _df: pd.DataFrame = pd.read_csv(filepath_or_buffer=data_set_file_path, sep=sep)
    _ml_sampler: MLSampler = MLSampler(df=_df,
                                       target=target_feature_name,
                                       features=features,
                                       train_size=train_size,
                                       random_sample=random_sample,
                                       stratification=False,
                                       seed=seed
                                       )
    _train_test_split: dict = _ml_sampler.train_test_sampling(validation_split=validation_size)
    _train_df: pd.DataFrame = _train_test_split.get('x_train')
    _train_df[target_feature_name] = _train_test_split.get('y_train')
    _train_data_set_path: str = os.path.join(output_file_path_train_test_split_data, 'train.csv')
    _train_df.to_csv(path_or_buf=_train_data_set_path, sep=sep, header=True, index=False)
    _test_df: pd.DataFrame = _train_test_split.get('x_test')
    _test_df[target_feature_name] = _train_test_split.get('y_test')
    _test_data_set_path: str = os.path.join(output_file_path_train_test_split_data, 'test.csv')
    _test_df.to_csv(path_or_buf=_test_data_set_path, sep=sep, header=True, index=False)
    _sampling_metadata: dict = dict(n_features=_train_df.shape[1] - 1,
                                    n_cases={'train': _train_df.shape[0],
                                             'test': _test_df.shape[0],
                                             }
                                    )
    if _train_test_split.get('x_val') is not None and _train_test_split.get('y_val') is not None:
        _val_df: pd.DataFrame = _train_test_split.get('x_val')
        _val_df[target_feature_name] = _train_test_split.get('y_val')
        _val_data_set_path: str = os.path.join(output_file_path_train_test_split_data, 'val.csv')
        _val_df.to_csv(path_or_buf=_val_data_set_path, sep=sep, header=True, index=False)
        _sampling_metadata['n_cases'].update({'val': _val_df.shape[0]})
    else:
        _val_data_set_path: str = None
    for file_path, obj in [(_train_data_set_path, _train_data_set_path),
                           (_test_data_set_path, _test_data_set_path),
                           (_val_data_set_path, _val_data_set_path),
                           (output_file_path_sampling_metadata, _sampling_metadata)
                           ]:
        with open(file_path, 'w') as _file:
            json.dump(obj, _file)
    if output_bucket_name is not None:
        _s3_resource: boto3 = boto3.resource('s3')
        _s3_obj: _s3_resource.Object = _s3_resource.Object(output_bucket_name, output_file_path_sampling_metadata)
        _s3_obj.put(Body=json.dumps(obj=_sampling_metadata))
    return [_train_data_set_path,
            _test_data_set_path,
            _val_data_set_path,
            _sampling_metadata
            ]
