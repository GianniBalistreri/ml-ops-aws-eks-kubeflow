"""

Task: ... (Function to run in container)

"""

import argparse
import random

from aws import file_exists, load_file_from_s3, save_file_to_s3
from datetime import datetime
from evolutionary_algorithm import (
    EVOLUTIONARY_ALGORITHMS, EvolutionaryAlgorithmException, adjust_swarm, check_for_stopping, generate_meta_data_template,
    mating_pool, select_best_individual
)
from file_handler import file_handler
from typing import Any, Dict, List, NamedTuple


PARSER = argparse.ArgumentParser(description="evolutionary algorithm")
PARSER.add_argument('-metadata_file_path', type=str, required=True, default=None, help='file path of the data set')
PARSER.add_argument('-algorithm', type=str, required=False, default='ga', help='assignment of features to analytical data types')
PARSER.add_argument('-max_generations', type=int, required=False, default=10, help='name of the S3 output bucket')
PARSER.add_argument('-pop_size', type=int, required=False, default=64, help='file path of the data health check output')
PARSER.add_argument('-burn_in_generations', type=int, required=False, default=-1, help='file path of features containing too much missing data')
PARSER.add_argument('-warm_start', type=bool, required=False, default=True, help='file path of invariant features')
PARSER.add_argument('-change_rate', type=float, required=False, default=0.1, help='file path of duplicated features')
PARSER.add_argument('-change_prob', type=float, required=False, default=0.85, help='file path of valid features')
PARSER.add_argument('-parents_ratio', type=float, required=False, default=0.5, help='file path of the proportion of valid features')
PARSER.add_argument('-early_stopping', type=int, required=False, default=0, help='column separator')
PARSER.add_argument('-stopping_reason', type=str, required=False, default=None, help='threshold to classify features as invalid based on the amount of missing values')
PARSER.add_argument('-convergence', type=bool, required=False, default=False, help='complete customized file path of the data health check output')
PARSER.add_argument('-convergence_measure', type=str, required=False, default='min', help='complete customized file path of the missing data output')
PARSER.add_argument('-timer_in_seconds', type=int, required=False, default=43200, help='file path of invariant features')
PARSER.add_argument('-target', type=str, required=True, default=None, help='file path of duplicated features')
PARSER.add_argument('-features', type=list, required=True, default=None, help='file path of duplicated features')
PARSER.add_argument('-data_set_path', type=str, required=False, default=None, help='file path of duplicated features')
PARSER.add_argument('-train_data_file_path', type=str, required=True, default=None, help='file path of duplicated features')
PARSER.add_argument('-test_data_file_path', type=str, required=True, default=None, help='file path of valid features')
PARSER.add_argument('-validation_data_file_path', type=str, required=False, default=None, help='file path of the proportion of valid features')
PARSER.add_argument('-re_populate', type=bool, required=False, default=False, help='complete customized file path of the data health check output')
PARSER.add_argument('-max_trials', type=int, required=False, default=2, help='complete customized file path of the missing data output')
PARSER.add_argument('-models', type=list, required=True, default=None, help='file path of invariant features')
PARSER.add_argument('-model_params', type=Any, required=False, default=None, help='file path of duplicated features')
PARSER.add_argument('-output_file_path_metadata', type=str, required=True, default=None, help='file path of valid features')
PARSER.add_argument('-output_file_path_stop', type=str, required=False, default=None, help='file path of the proportion of valid features')
PARSER.add_argument('-output_file_path_stopping_reason', type=str, required=False, default=None, help='complete customized file path of the data health check output')
PARSER.add_argument('-output_file_path_generator_instructions', type=str, required=False, default=None, help='complete customized file path of the missing data output')
PARSER.add_argument('-output_file_path_current_iteration_meta_data', type=str, required=False, default=None, help='file path of invariant features')
PARSER.add_argument('-output_file_path_iteration_history', type=str, required=False, default=None, help='file path of duplicated features')
PARSER.add_argument('-output_file_path_evolution_history', type=str, required=False, default=None, help='file path of valid features')
PARSER.add_argument('-output_file_path_evolution_gradient', type=str, required=False, default=None, help='file path of the proportion of valid features')
ARGS = PARSER.parse_args()


def evolutionary_algorithm(metadata_file_path: str,
                           target: str,
                           features: List[str],
                           models: List[str],
                           train_data_file_path: str,
                           test_data_file_path: str,
                           output_file_path_metadata: str,
                           output_file_path_evolve: str,
                           output_file_path_stopping_reason: str,
                           output_file_path_generator_instructions: str,
                           output_file_path_current_iteration_meta_data: str,
                           output_file_path_iteration_history: str,
                           output_file_path_evolution_history: str,
                           output_file_path_evolution_gradient: str,
                           validation_data_file_path: str = None,
                           algorithm: str = 'ga',
                           max_generations: int = 10,
                           pop_size: int = 64,
                           burn_in_generations: int = -1,
                           warm_start: bool = True,
                           change_rate: float = 0.1,
                           change_prob: float = 0.85,
                           parents_ratio: float = 0.5,
                           early_stopping: int = 0,
                           stopping_reason: str = None,
                           convergence: bool = False,
                           convergence_measure: str = 'min',
                           timer_in_seconds: int = 43200,
                           re_populate: bool = False,
                           max_trials: int = 2,
                           model_params: dict = None
                           ) -> NamedTuple('outputs', [('metadata', dict),
                                                       ('evolve', str),
                                                       ('stopping_reason', str),
                                                       ('generator_instructions', dict),
                                                       ('current_generation_meta_data', dict),
                                                       ('generation_history', dict),
                                                       ('evolution_history', dict),
                                                       ('evolution_gradient', dict)
                                                       ]
                                           ):
    """
    Optimize machine learning models

    :param metadata_file_path:
    :param target:
    :param features:
    :param models:
    :param train_data_file_path:
    :param test_data_file_path:
    :param output_file_path_metadata:
    :param output_file_path_evolve:
    :param output_file_path_stopping_reason:
    :param output_file_path_generator_instructions:
    :param output_file_path_current_iteration_meta_data:
    :param output_file_path_iteration_history:
    :param output_file_path_evolution_history:
    :param output_file_path_evolution_gradient:
    :param validation_data_file_path:
    :param algorithm:
    :param max_generations:
    :param pop_size:
    :param burn_in_generations:
    :param warm_start:
    :param change_rate:
    :param change_prob:
    :param parents_ratio:
    :param early_stopping:
    :param stopping_reason:
    :param convergence:
    :param convergence_measure:
    :param timer_in_seconds:
    :param re_populate:
    :param max_trials:
    :param model_params:
    :return: NamedTuple

    """
    if file_exists(file_path=metadata_file_path):
        _metadata: dict = load_file_from_s3(file_path=metadata_file_path)
        # TODO: gather population data
        _metadata['iteration_history']['population']
        _metadata['iteration_history']['inheritance']
        _elapsed_time_in_seconds_of_previous_iteration: int = (datetime.now() - _metadata['start_time']).seconds
        _metadata['iteration_history']['time'].append(_elapsed_time_in_seconds_of_previous_iteration)
        # TODO: gather evolution history data
        # TODO: gather evolution gradient data
        _check_for_stopping: dict = check_for_stopping(metadata=_metadata)
        _stopping_reason: str = _check_for_stopping.get('stopping_reason')
        if _check_for_stopping.get('evolve'):
            _evolve: str = 'true'
            _metadata['current_iteration'] += 1
            if _metadata['algorithm'] == 'random':
                _algorithm: str = random.choice(EVOLUTIONARY_ALGORITHMS)
            elif _metadata['algorithm'] in ['ga_si', 'si_ga']:
                if _metadata['current_iteration_algorithm'][-1] == 'ga':
                    _algorithm: str = 'si'
                else:
                    _algorithm: str = 'ga'
            else:
                _algorithm: str = _metadata['algorithm']
            _metadata['current_iteration_algorithm'].append(_algorithm)
            if _algorithm == 'ga':
                _mating_pool = mating_pool(metadata=_metadata)
                _metadata['generated_individuals'].extend(_mating_pool.get('new_ids'))
                _generator_instructions: List[dict] = _mating_pool.get('mutations')
            elif _algorithm == 'si':
                _best_idx: Dict[str, int] = select_best_individual(fitness_scores=_metadata['current_iteration_meta_data']['fitness_score'])
                _metadata['best_local_idx'] = _best_idx.get('best_local_idx')
                _metadata['best_global_idx'] = _best_idx.get('best_global_idx')
                _swarm_adjustments: dict = adjust_swarm(metadata=_metadata)
                _metadata['generated_individuals'].extend(_swarm_adjustments.get('new_ids'))
                _generator_instructions: List[dict] = _swarm_adjustments.get('adjustments')
            else:
                raise EvolutionaryAlgorithmException(f'Evolutionary algorithm ({_algorithm}) not supported')
        else:
            _evolve: str = 'false'
            _generator_instructions: List[dict] = []
            _metadata['end_time'] = datetime.now()
    else:
        _metadata: dict = generate_meta_data_template()
        _evolve: str = 'false'
        _stopping_reason: str = None
        _generator_instructions: List[dict] = []
    _current_iteration_meta_data: dict = _metadata['current_iteration_meta_data']
    _iteration_history: dict = _metadata['iteration_history']
    _evolution_history: dict = _metadata['evolution_history']
    _evolution_gradient: dict = _metadata['evolution_gradient']
    for file_path, obj in [(output_file_path_metadata, _metadata),
                           (output_file_path_evolve, _evolve),
                           (output_file_path_stopping_reason, _stopping_reason),
                           (output_file_path_generator_instructions, _generator_instructions),
                           (output_file_path_current_iteration_meta_data, _current_iteration_meta_data),
                           (output_file_path_iteration_history, _iteration_history),
                           (output_file_path_evolution_history, _evolution_history),
                           (output_file_path_evolution_gradient, _evolution_gradient)
                           ]:
        file_handler(file_path=file_path, obj=obj)
    return [_metadata,
            _evolve,
            _stopping_reason,
            _current_iteration_meta_data,
            _iteration_history,
            _evolution_history,
            _evolution_gradient
            ]


if __name__ == '__main__':
    evolutionary_algorithm(data_set_path=ARGS.data_set_path,
                           analytical_data_types=ARGS.analytical_data_types,
                           output_file_path_data_health_check=ARGS.output_file_path_data_health_check,
                           output_file_path_missing_data=ARGS.output_file_path_missing_data,
                           output_file_path_invariant=ARGS.output_file_path_invariant,
                           output_file_path_duplicated=ARGS.output_file_path_duplicated,
                           output_file_path_valid_features=ARGS.output_file_path_valid_features,
                           output_file_path_prop_valid_features=ARGS.output_file_path_prop_valid_features,
                           sep=ARGS.sep,
                           missing_value_threshold=ARGS.missing_value_threshold,
                           output_file_path_data_health_check_customized=ARGS.output_file_path_data_health_check_customized,
                           output_file_path_missing_data_customized=ARGS.output_file_path_missing_data_customized,
                           output_file_path_invariant_customized=ARGS.output_file_path_invariant_customized,
                           output_file_path_duplicated_customized=ARGS.output_file_path_duplicated_customized,
                           output_file_path_valid_features_customized=ARGS.output_file_path_valid_features_customized,
                           output_file_path_prop_valid_features_customized=ARGS.output_file_path_prop_valid_features_customized
                           )
