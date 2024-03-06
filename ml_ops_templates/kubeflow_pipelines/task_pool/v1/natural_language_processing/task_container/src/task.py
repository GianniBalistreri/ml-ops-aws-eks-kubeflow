"""

Task: ... (Function to run in container)

"""

import boto3
import json
import os
import numpy as np
import pandas as pd
import pickle

from datetime import datetime
from natural_language_processing import TextMiner, TextMinerException
from resource_metrics import get_available_cpu, get_cpu_utilization, get_cpu_utilization_per_core, get_memory, get_memory_utilization
from typing import NamedTuple, List


def natural_language_processing(data_set_path: str,
                                action: str,
                                output_data_path: str,
                                sep: str = '\t',
                                target_feature: str = 'text',
                                lang: str = None,
                                lang_model: str = 'spacy',
                                lang_model_size: str = 'sm',
                                lang_model_framework: str = 'spacy',
                                segmentation_threshold: float = 0.5,
                                auto_interpret_natural_language: bool = False,
                                numbers: bool = True,
                                stop_words: bool = True,
                                special_characters: bool = True,
                                punct: bool = True,
                                pron: bool = True,
                                entity: bool = True,
                                lemmatizing: bool = True
                                ) -> NamedTuple('outputs', [('cleaned_data_set_path', str)]):
    """
    Process natural language and text like data

    :param data_set_path: str
        Complete file path of the data set

    :param action: str
        Name of the action to take
            -> clean: Clean text

    :param output_data_path: str
        Path of the processed data set to save

    :param sep: str
        Separator

    :param target_feature: str
        Name of the target text feature

    :param lang: str
            Language of the text (use multi-language framework if lang is None)

    :param lang_model: str
        Name of the language model to use

    :param lang_model_size: str
        Name of the model size type:
            -> sm, small -> small (pre-trained) language model
            -> lg, large, big -> large (pre-trained) language model

    :param lang_model_framework: str
        Name of the language model framework

    :param segmentation_threshold: float
        Threshold for identify certain segments

    :param auto_interpret_natural_language: bool
        Whether to interpret natural language automatically while initialization

    :param numbers: bool
        Whether to remove numbers from text or not

    :param stop_words: bool
        Whether to remove stop-words from text or not

    :param special_characters: bool
        Whether to remove special characters from text or not

    :param punct: bool
        Whether to remove punctuation from text or not

    :param pron: bool
        Whether to remove pronouns from text or not

    :param entity: bool
        Whether to remove recognized entities from text or not

    :param lemmatizing: bool
        Whether to lemmatize (trim words to their word-stem) text or not

    :return: NamedTuple
        Path of the cleaned data set
    """
    _cpu_available: int = get_available_cpu(logging=True)
    _memory_total: float = get_memory(total=True, logging=True)
    _memory_available: float = get_memory(total=False, logging=True)
    _df: pd.DataFrame = pd.read_csv(filepath_or_buffer=data_set_path, sep=sep)
    _text_miner: TextMiner = TextMiner(df=_df,
                                       features=_df.columns.tolist(),
                                       lang=lang,
                                       lang_model=lang_model,
                                       lang_model_size=lang_model_size,
                                       lang_model_framework=lang_model_framework,
                                       segmentation_threshold=segmentation_threshold,
                                       auto_interpret_natural_language=auto_interpret_natural_language
                                       )
    if action == 'clean':
        _clean_phrases: List[str] = []
        for phrase in _df[target_feature].values:
            _clean_phrases.append(_text_miner.clean_text(phrase=phrase,
                                                         numbers=numbers,
                                                         stop_words=stop_words,
                                                         special_characters=special_characters,
                                                         punct=punct,
                                                         pron=pron,
                                                         entity=entity,
                                                         lemmatizing=lemmatizing
                                                         )
                                  )
        _df['text_cleaned'] = _clean_phrases
        _df.to_csv(path_or_buf=output_data_path, index=False, sep=sep)
    else:
        raise TextMinerException(f'Action ({action}) not supported')
    _cpu_utilization: float = get_cpu_utilization(interval=1, logging=True)
    _cpu_utilization_per_cpu: List[float] = get_cpu_utilization_per_core(interval=1, logging=True)
    _memory_utilization: float = get_memory_utilization(logging=True)
    _memory_available = get_memory(total=False, logging=True)
    return [output_data_path]
