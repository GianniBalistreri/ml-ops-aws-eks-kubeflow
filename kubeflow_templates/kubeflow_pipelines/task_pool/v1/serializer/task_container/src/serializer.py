"""

Prepare data for processing in parallel

"""

import pandas as pd

from aws import load_file_from_s3
from custom_logger import Log
from typing import List


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
                 sep: str = ','
                 ):
        """
        :param file_paths: List[str]
        Complete file paths to serialize

        :param output_file_path: str
            Complete file path of the output data set

        :param contents: List[dict]
            List of evolutionary algorithm parallelization contents

        :param sep: str
            Separator
        """
        self.file_paths: List[str] = file_paths
        self.output_file_path: str = output_file_path
        self.contents: List[dict] = contents
        self.sep: str = sep

    def _serialize_cases(self) -> None:
        """
        Serialize cases of different files and container
        """
        _df: pd.DataFrame = pd.DataFrame()
        for file_path in self.file_paths:
            _df_chunk: pd.DataFrame = pd.read_csv(filepath_or_buffer=file_path, sep=self.sep)
            _df = pd.concat(objs=[_df, _df_chunk], axis=0)
        _df.to_csv(path_or_buf=self.output_file_path, sep=self.sep, header=True, index=False)
        Log().log(msg=f'Serialize {_df.shape[0]} cases from {len(self.file_paths)} chunks')

    def _serialize_features(self) -> None:
        """
        Serialize features of different files and container
        """
        _df: pd.DataFrame = pd.DataFrame()
        for file_path in self.file_paths:
            _df_chunk: pd.DataFrame = pd.read_csv(filepath_or_buffer=file_path, sep=self.sep)
            _df = pd.concat(objs=[_df, _df_chunk], axis=1)
        _df.to_csv(path_or_buf=self.output_file_path, sep=self.sep, header=True, index=False)
        Log().log(msg=f'Serialize {_df.shape[0]} features from {len(self.file_paths)} chunks')

    def _serialize_evolutionary_results(self) -> dict:
        """
        Serialize json file from different json file

        :return: dict
            Serialized json file content
        """
        _serialized_contents: dict = {}
        for content in self.contents:
            _key: str = list(content.keys())[0]
            _param_file_path: str = content[_key]['param_file_path']
            _eval_metric_file_path: str = content[_key]['eval_metric_file_path']
            _param: dict = load_file_from_s3(file_path=_param_file_path)
            _eval_metric: dict = load_file_from_s3(file_path=_eval_metric_file_path)
            _serialized_contents.update({_key: content[_key]})
            _serialized_contents[_key].update({'parameter': _param, 'metric': _eval_metric})
        Log().log(msg=f'Serialize dictionary from {len(self.contents)} chunks')
        return _serialized_contents

    def main(self, action: str) -> dict:
        """
        Serialize elements

        :param action: str
            Name of the parallelization action
                -> cases: serialize cases of given data set
                -> features: serialize features of given data set
                -> evolutionary_algorithm: serialize hyperparameter settings and evaluation metrics
        """
        _serialization: dict = {}
        if action == 'cases':
            self._serialize_cases()
        elif action == 'features':
            self._serialize_features()
        elif action == 'evolutionary_algorithm':
            _serialization = self._serialize_evolutionary_results()
        else:
            raise SerializerException(f'Action ({action}) not supported')
        return _serialization
