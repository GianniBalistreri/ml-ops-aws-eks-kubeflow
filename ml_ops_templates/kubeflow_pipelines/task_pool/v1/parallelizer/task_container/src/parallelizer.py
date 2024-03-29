"""

Prepare data for processing in parallel

"""

import boto3
import numpy as np
import pandas as pd

from aws import load_file_from_s3_as_df, save_file_to_s3_as_df
from custom_logger import Log
from typing import Dict, List, Union

MAX_CHUNKS: int = 100


class ParallelizerException(Exception):
    """
    Class for handling exceptions for class Parallelizer
    """
    pass


class Parallelizer:
    """
    Class for parallelize elements
    """
    def __init__(self,
                 file_path: str,
                 chunks: int,
                 persist_data: bool = True,
                 analytical_data_types: Dict[str, List[str]] = None,
                 elements: Union[List[str], np.array] = None,
                 split_by: str = None,
                 s3_bucket_name: str = None,
                 prefix: str = None,
                 sep: str = ','
                 ):
        """
        :param file_path: str
            Complete path to get file paths to distribute

        :param chunks: int
            Number of chunks to distribute

        :param persist_data: bool
            Whether to persist distributed data in several data set or not

        :param analytical_data_types: Dict[str, List[str]]
            Assigned analytical data type for each feature

        :param elements: Union[List[str], np.array]
            List or array of elements to distribute

        :param split_by: str
            Name of the features to split cases by

        :param s3_bucket_name: str
            Name of the S3 bucket

        :param prefix: str
            Prefix to filter by (e.g. dir/file_name)

        :param sep: str
            Separator
        """
        self.file_path: str = file_path
        self.chunks: int = chunks
        self.persist_data: bool = persist_data
        self.analytical_data_types: Dict[str, List[str]] = analytical_data_types
        self.elements: Union[List[str], np.array] = elements
        self.split_by: str = split_by
        self.s3_bucket_name: str = s3_bucket_name
        self.prefix: str = prefix
        self.sep: str = sep

    def _adjust_number_of_chunks(self, n_available_chunks: int) -> None:
        """
        Adjust number of chunks if it reaches any limit

        :param n_available_chunks: int
            Number of available chunks
        """
        if self.chunks > MAX_CHUNKS:
            Log().log(msg=f'Reduce number of chunks from {self.chunks} to {MAX_CHUNKS} (maximum allowed)')
            self.chunks = MAX_CHUNKS
        if self.chunks > n_available_chunks:
            Log().log(msg=f'Reduce number of chunks from {self.chunks} to {n_available_chunks} (available)')
            self.chunks = n_available_chunks

    def distribute_analytical_data_types(self) -> list:
        """
        Distribute features by analytical data type assignment

        :return: list
            Distributed features based on analytical data type assignment
        """
        _df: pd.DataFrame = load_file_from_s3_as_df(file_path=self.file_path, sep=self.sep)
        Log().log(msg=f'Load data set: {self.file_path} -> Cases={_df.shape[0]}, Features={_df.shape[1]}')
        _pairs: list = []
        for analytical_data_type in self.analytical_data_types:
            if len(analytical_data_type) == 0:
                continue
            if self.persist_data:
                _new_file_name: str = self.file_path.replace('.', f'_{analytical_data_type}.')
                _df_subset: pd.DataFrame = _df.loc[:, self.analytical_data_types.get(analytical_data_type)]
                save_file_to_s3_as_df(file_path=_new_file_name, df=_df_subset, sep=self.sep)
                Log().log(msg=f'Save data set: {_new_file_name} -> Cases={_df_subset.shape[0]}, Features={_df_subset.shape[1]}')
                _pairs.append(_new_file_name)
            else:
                _pairs.append(self.analytical_data_types.get(analytical_data_type))
            Log().log(msg=f'Select {len(analytical_data_type)} features assign to analytical data type {analytical_data_type} for {len(_pairs)}. chunk')
        Log().log(msg=f'Distribute {_df.shape[1]} features into {len(_pairs)} chunks')
        return _pairs

    def distribute_cases(self) -> list:
        """
        Distribute cases to different files and container

        :return: list
            Distributed cases
        """
        _df: pd.DataFrame = load_file_from_s3_as_df(file_path=self.file_path, sep=self.sep)
        Log().log(msg=f'Load data set: {self.file_path} -> Cases={_df.shape[0]}, Features={_df.shape[1]}')
        _pairs: list = []
        if self.split_by is None:
            self._adjust_number_of_chunks(n_available_chunks=_df.shape[0])
            _pairs_array: List[np.array] = np.array_split(ary=_df.index.values, indices_or_sections=self.chunks)
            for i, pair in enumerate(_pairs_array):
                if self.persist_data:
                    _new_file_name: str = self.file_path.replace('.', f'_{i}.')
                    _df_subset: pd.DataFrame = _df.iloc[pair.tolist(), :]
                    save_file_to_s3_as_df(file_path=_new_file_name, df=_df_subset, sep=self.sep)
                    Log().log(msg=f'Save data set: {_new_file_name} -> Cases={_df_subset.shape[0]}, Features={_df_subset.shape[1]}')
                    _pairs.append(_new_file_name)
                else:
                    _pairs.append(pair.tolist())
                Log().log(msg=f'Select {len(pair.tolist())} cases for {len(_pairs)}. chunk')
            Log().log(msg=f'Distributed {_df.shape[0]} cases into {self.chunks} chunks')
        else:
            for value in _df[self.split_by].unique():
                _new_file_name: str = self.file_path.replace('.', f'_{value}.')
                _df_subset: pd.DataFrame = _df.loc[_df[self.split_by] == value, :]
                save_file_to_s3_as_df(file_path=_new_file_name, df=_df_subset, sep=self.sep)
                Log().log(msg=f'Save data set: {_new_file_name} -> Cases={_df_subset.shape[0]}, Features={_df_subset.shape[1]}')
                _pairs.append(_new_file_name)
        return _pairs

    def distribute_elements(self) -> List[list]:
        """
        Distribute given list elements to different container

        :return: List[list]
            Distributed list elements
        """
        if isinstance(self.elements, list):
            _array: np.array = np.array(self.elements)
        else:
            _array: np.array = self.elements
        self._adjust_number_of_chunks(n_available_chunks=len(self.elements))
        _pairs_array: List[np.array] = np.array_split(ary=_array, indices_or_sections=self.chunks)
        _pairs: list = []
        for pair in _pairs_array:
            _pairs.append(pair.tolist())
            Log().log(msg=f'Select {len(pair.tolist())} elements for {len(_pairs)}. chunk')
        Log().log(msg=f'Distributed {len(self.elements)} elements into {self.chunks} chunks')
        return _pairs

    def distribute_features(self) -> list:
        """
        Distribute features to different files and container

        :return: list
            Distributed features
        """
        _df: pd.DataFrame = load_file_from_s3_as_df(file_path=self.file_path, sep=self.sep)
        Log().log(msg=f'Load data set: {self.file_path} -> Cases={_df.shape[0]}, Features={_df.shape[1]}')
        self._adjust_number_of_chunks(n_available_chunks=_df.shape[1])
        _pairs_array: List[np.array] = np.array_split(ary=_df.columns.values, indices_or_sections=self.chunks)
        _pairs: list = []
        for i, pair in enumerate(_pairs_array):
            if self.persist_data:
                _new_file_name: str = self.file_path.replace('.', f'_{i}.')
                _df_subset: pd.DataFrame = _df.loc[:, pair.tolist()]
                save_file_to_s3_as_df(file_path=_new_file_name, df=_df_subset, sep=self.sep)
                Log().log(msg=f'Save data set: {_new_file_name} -> Cases={_df_subset.shape[0]}, Features={_df_subset.shape[1]}')
                _pairs.append(_new_file_name)
            else:
                _pairs.append(pair.tolist())
            Log().log(msg=f'Select {len(pair.tolist())} features for {len(_pairs)}. chunk')
        Log().log(msg=f'Distribute {_df.shape[1]} features into {self.chunks} chunks')
        return _pairs

    def distribute_file_paths(self) -> List[List[str]]:
        """
        Distribute file paths to different container

        :return: List[List[str]]
            Distributed file paths
        """
        _s3_resource: boto3 = boto3.resource('s3')
        _paginator = _s3_resource.get_paginator('list_objects')
        _operation_parameters: Dict[str, str] = {'Bucket': self.s3_bucket_name}
        if self.prefix is not None:
            _operation_parameters.update({'Prefix': self.prefix})
        _page_iterator = _paginator.paginate(**_operation_parameters)
        _file_names: List[str] = []
        for page in _page_iterator:
            for content in page['Contents']:
                _file_names.append(content['Key'])
        self._adjust_number_of_chunks(n_available_chunks=len(_file_names))
        _pairs_array: List[np.array] = np.array_split(ary=np.array(_file_names), indices_or_sections=self.chunks)
        _pairs: List[List[str]] = []
        for pair in _pairs_array:
            _pairs.append(pair.tolist())
            Log().log(msg=f'Select {len(pair.tolist())} files for {len(_pairs)}. chunk')
        Log().log(msg=f'Distribute {len(_file_names)} files into {self.chunks} chunks')
        return _pairs
