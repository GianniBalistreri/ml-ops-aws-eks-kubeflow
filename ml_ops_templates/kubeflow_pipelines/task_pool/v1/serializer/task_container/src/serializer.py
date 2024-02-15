"""

Prepare data for processing in parallel

"""

import copy
import pandas as pd

from aws import load_file_from_s3, load_file_from_s3_as_df, save_file_to_s3, save_file_to_s3_as_df
from custom_logger import Log
from typing import Dict,List


class SerializerException(Exception):
    """
    Class for handling exceptions for class Serializer
    """
    pass


class Serializer:
    """
    Class for serialize parallelized elements
    """
    def __init__(self,
                 file_paths: List[str],
                 output_file_path: str,
                 contents: List[dict],
                 label_feature_name: str = None,
                 labels: list = None,
                 sep: str = ','
                 ):
        """
        :param file_paths: List[str]
        Complete file paths to serialize

        :param output_file_path: str
            Complete file path of the output data set

        :param contents: List[dict]
            List of evolutionary algorithm parallelization contents

        :param label_feature_name: str
            Name of the label feature that contains given labels

        :param labels: list
            Labels used in to identify origin of the cases to serialize

        :param sep: str
            Separator
        """
        self.file_paths: List[str] = file_paths
        self.output_file_path: str = output_file_path
        self.contents: List[dict] = contents
        if labels is not None:
            if len(file_paths) != len(labels):
                raise SerializerException(f'Number of labels ({len(labels)}) not equal to number of files ({len(file_paths)})')
        self.label_feature_name: str = label_feature_name
        self.labels: list = labels
        self.sep: str = sep

    def serialize_analytical_data_types(self) -> None:
        """
        Serialize analytical data types of different files and container
        """
        _analytical_data_types: Dict[str, List[str]] = {}
        for file_path in self.file_paths:
            _analytical_data_types_chunk: Dict[str, List[str]] = load_file_from_s3(file_path=file_path)
            Log().log(msg=f'Load chunked analytical data types: {file_path}')
            for analytical_data_type in _analytical_data_types_chunk.keys():
                if analytical_data_type in _analytical_data_types.keys():
                    _analytical_data_types[analytical_data_type].extend(_analytical_data_types_chunk[analytical_data_type])
                else:
                    _analytical_data_types.update({analytical_data_type: _analytical_data_types_chunk[analytical_data_type]})
                _analytical_data_types[analytical_data_type] = copy.deepcopy(list(set(_analytical_data_types[analytical_data_type])))
        save_file_to_s3(file_path=self.output_file_path, obj=_analytical_data_types)
        Log().log(msg=f'Serialize analytical data types from {len(self.file_paths)} chunks')
        Log().log(msg=f'Save serialized analytical data types: {self.output_file_path}')

    def serialize_cases(self) -> None:
        """
        Serialize cases of different files and container
        """
        _df: pd.DataFrame = pd.DataFrame()
        for i, file_path in enumerate(self.file_paths):
            _df_chunk: pd.DataFrame = load_file_from_s3_as_df(file_path=file_path, sep=self.sep)
            Log().log(msg=f'Load chunked data set: {file_path} -> Cases={_df_chunk.shape[0]}, Features={_df_chunk.shape[1]}')
            if self.labels is not None:
                _df_chunk[self.label_feature_name] = self.labels[i]
                Log().log(msg=f'Generate label feature to identify origin of chunk {i}')
            _df = pd.concat(objs=[_df, _df_chunk], axis=0)
        _df.index = [i for i in range(0, _df.shape[0], 1)]
        save_file_to_s3_as_df(file_path=self.output_file_path, df=_df, sep=self.sep)
        Log().log(msg=f'Serialize {_df.shape[0]} cases from {len(self.file_paths)} chunks')
        Log().log(msg=f'Save serialized data set: {self.output_file_path}')

    def serialize_data_health_check_results(self) -> dict:
        """
        Serialize data health check results of different files and container

        :return dict
            Serialized data health check result output
        """
        _data_health_check: dict = {}
        _n_features: int = 0
        for file_path in self.file_paths:
            _data_health_check_chunk: dict = load_file_from_s3(file_path=file_path)
            Log().log(msg=f'Load chunked data health check results: {file_path}')
            for health_check in _data_health_check_chunk.keys():
                if isinstance(_data_health_check_chunk[health_check], list):
                    if health_check in _data_health_check.keys():
                        _data_health_check[health_check].extend(_data_health_check_chunk[health_check])
                    else:
                        _data_health_check.update({health_check: _data_health_check_chunk[health_check]})
            _n_features += (_data_health_check_chunk['n_valid_features'] / _data_health_check_chunk['prop_valid_features']) * 100
        _data_health_check.update({'n_valid_features': len(_data_health_check['valid_features']),
                                   'prop_valid_features': round(len(_data_health_check['valid_features']) / _n_features)
                                   })
        save_file_to_s3(file_path=self.output_file_path, obj=_data_health_check)
        Log().log(msg=f'Serialize data health check results from {len(self.file_paths)} chunks')
        Log().log(msg=f'Save serialized data health check: {self.output_file_path}')
        return _data_health_check

    def serialize_features(self) -> List[str]:
        """
        Serialize features of different files and container

        :return List[str]
            Names of the features
        """
        _df: pd.DataFrame = pd.DataFrame()
        _features: List[str] = []
        for file_path in self.file_paths:
            _df_chunk: pd.DataFrame = load_file_from_s3_as_df(file_path=file_path, sep=self.sep)
            _features.extend(_df_chunk.columns.tolist())
            Log().log(msg=f'Load chunked data set: {file_path} -> Cases={_df_chunk.shape[0]}, Features={_df_chunk.shape[1]}')
            _df = pd.concat(objs=[_df, _df_chunk], axis=1)
        save_file_to_s3_as_df(file_path=self.output_file_path, df=_df, sep=self.sep)
        Log().log(msg=f'Serialize {_df.shape[1]} features from {len(self.file_paths)} chunks')
        Log().log(msg=f'Save serialized data set: {self.output_file_path}')
        return _features

    def serialize_evolutionary_results(self) -> None:
        """
        Serialize json file from different json file
        """
        _serialized_contents: dict = {}
        _generator_instruction: List[dict] = load_file_from_s3(file_path=self.file_paths[0])
        for i, individual in enumerate(_generator_instruction):
            if individual.get('id') is None:
                continue
            _param: dict = load_file_from_s3(file_path=individual.get('model_param_path'))
            Log().log(msg=f'Load model hyperparameter for individual {i}: {individual.get("model_param_path")}')
            _eval_metric: dict = load_file_from_s3(file_path=individual.get('model_fitness_path'))
            Log().log(msg=f'Load evaluation metric: {individual.get("model_fitness_path")}')
            _metadata: dict = load_file_from_s3(file_path=individual.get('model_metadata_path'))
            Log().log(msg=f'Load model metadata: {individual.get("model_metadata_path")}')
            _serialized_contents.update({str(i): dict(id=individual.get('id'),
                                                      model_name=individual.get('model_name'),
                                                      param=_param,
                                                      param_changed=_metadata.get('param_changed'),
                                                      fitness_metric=_eval_metric['test'][list(_eval_metric['test'].keys())[0]],
                                                      fitness_score=_eval_metric.get('sml_score'),
                                                      parent=individual.get('parent'),
                                                      change_type=individual.get('change_type'),
                                                      train_test_diff=_eval_metric['train'][list(_eval_metric['train'].keys())[0]] - _eval_metric['test'][list(_eval_metric['test'].keys())[0]],
                                                      train_time_in_seconds=_metadata.get('train_time_in_sec'),
                                                      original_ml_train_metric=_eval_metric['train'][list(_eval_metric['train'].keys())[0]],
                                                      original_ml_test_metric=_eval_metric['test'][list(_eval_metric['test'].keys())[0]]
                                                      )
                                         })
        save_file_to_s3(file_path=self.output_file_path, obj=_serialized_contents)
        Log().log(msg=f'Serialize environment reaction from {len(_generator_instruction)} chunks')
        Log().log(msg=f'Save serialized environment reaction: {self.output_file_path}')

    def serialize_processor_memory(self) -> dict:
        """
        Serialize processor memory of the feature engineer

        :return dict
            Serialized processor memory
        """
        _serialized_processor_memory: dict = {}
        for i, file_path in enumerate(self.file_paths):
            _processor_memory_chunk: dict = load_file_from_s3(file_path=file_path)
            Log().log(msg=f'Load chunked processing memory: {file_path}')
            if i == 0:
                _serialized_processor_memory.update({'level': _processor_memory_chunk['level'],
                                                     'processor': _processor_memory_chunk['processor'],
                                                     'feature_relations': _processor_memory_chunk['feature_relations'],
                                                     'analytical_data_types': _processor_memory_chunk['analytical_data_types'],
                                                     'next_level_numeric_features_base': _processor_memory_chunk['next_level_numeric_features_base'],
                                                     'next_level_categorical_features_base': _processor_memory_chunk['next_level_categorical_features_base'],
                                                     'new_target_feature': _processor_memory_chunk['new_target_feature']
                                                     })
            else:
                for level in _processor_memory_chunk['level'].keys():
                    if _serialized_processor_memory['level'].get(level) is None:
                        _serialized_processor_memory['level'].update({level: _processor_memory_chunk['level'][level]})
                    else:
                        _serialized_processor_memory['level'][level].extend(_processor_memory_chunk['level'][level])
                for processor in _processor_memory_chunk['processor'].keys():
                    if _serialized_processor_memory['processor'].get(processor) is None:
                        _serialized_processor_memory['processor'].update({processor: _processor_memory_chunk['processor'][processor]})
                for feature_relation in _processor_memory_chunk['feature_relations'].keys():
                    if _serialized_processor_memory['feature_relations'].get(feature_relation) is None:
                        _serialized_processor_memory['feature_relations'].update({feature_relation: _processor_memory_chunk['feature_relations'][feature_relation]})
                for analytical_data_type in _processor_memory_chunk['analytical_data_types'].keys():
                    if _serialized_processor_memory['analytical_data_types'].get(analytical_data_type) is None:
                        _serialized_processor_memory['analytical_data_types'].update({analytical_data_type: _processor_memory_chunk['analytical_data_types'][analytical_data_type]})
                    else:
                        _serialized_processor_memory['analytical_data_types'][analytical_data_type].extend(_processor_memory_chunk['analytical_data_types'][analytical_data_type])
                _serialized_processor_memory['next_level_numeric_features_base'].extend(_processor_memory_chunk['next_level_numeric_features_base'])
                _serialized_processor_memory['next_level_categorical_features_base'].extend(_processor_memory_chunk['next_level_categorical_features_base'])
                _serialized_processor_memory['new_target_feature'] = _processor_memory_chunk['new_target_feature']
                _serialized_processor_memory['predictors'].extend(_processor_memory_chunk['predictors'])
        save_file_to_s3(file_path=self.output_file_path, obj=_serialized_processor_memory)
        Log().log(msg=f'Serialize processor memory from {len(self.file_paths)} chunks')
        Log().log(msg=f'Save serialized processor memory: {self.output_file_path}')
        return _serialized_processor_memory
