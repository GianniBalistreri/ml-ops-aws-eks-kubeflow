"""

Task: ... (Function to run in container)

"""

import argparse
import copy
import os

from aws import file_exists, load_file_from_s3, save_file_to_s3
from custom_logger import Log
from evolutionary_algorithm import EvolutionaryAlgorithm
from file_handler import file_handler
from typing import Any, List, NamedTuple


PARSER = argparse.ArgumentParser(description="evolutionary algorithm")
PARSER.add_argument('-s3_metadata_file_path', type=str, required=True, default=None, help='S3 file path of the evolutionary algorithm metadata')
PARSER.add_argument('-target', type=str, required=True, default=None, help='name of the target feature')
PARSER.add_argument('-features', type=list, required=True, default=None, help='name of the features')
PARSER.add_argument('-models', type=list, required=True, default=None, help='name of the machine learning models')
PARSER.add_argument('-train_data_file_path', type=str, required=True, default=None, help='complete file path of the training data set')
PARSER.add_argument('-test_data_file_path', type=str, required=True, default=None, help='complete file path of the test data set')
PARSER.add_argument('-val_data_file_path', type=str, required=False, default=None, help='complete file path of the validation data set')
PARSER.add_argument('-algorithm', type=str, required=False, default='ga', help='name of the evolutionary algorithm')
PARSER.add_argument('-max_iterations', type=int, required=False, default=10, help='maximum number of iteration')
PARSER.add_argument('-pop_size', type=int, required=False, default=64, help='size of the population')
PARSER.add_argument('-burn_in_iterations', type=int, required=False, default=-1, help='number of iterations before early stopping methods are applied')
PARSER.add_argument('-warm_start', type=int, required=False, default=1, help='whether to use warm start or not')
PARSER.add_argument('-change_rate', type=float, required=False, default=0.1, help='rate of individual changes of characteristics')
PARSER.add_argument('-change_prob', type=float, required=False, default=0.85, help='probability for complete random individual change')
PARSER.add_argument('-parents_ratio', type=float, required=False, default=0.5, help='ratio of parents to define')
PARSER.add_argument('-early_stopping', type=int, required=False, default=0, help='')
PARSER.add_argument('-convergence', type=int, required=False, default=0, help='whether to check convergence of the population or not')
PARSER.add_argument('-convergence_measure', type=str, required=False, default='min', help='measurement for convergence')
PARSER.add_argument('-timer_in_seconds', type=int, required=False, default=43200, help='maximum number of seconds for evolution')
PARSER.add_argument('-re_populate', type=int, required=False, default=1, help='whether to re-populate initial population because of poor individual fitness score')
PARSER.add_argument('-re_populate_threshold', type=float, required=False, default=3.0, help='fitness score threshold for re-population')
PARSER.add_argument('-max_trials', type=int, required=False, default=2, help='number of trials for re-population before continuing evolution process')
PARSER.add_argument('-environment_reaction', type=Any, required=False, default=None, help='action of the generator and the according reactions of the environment')
PARSER.add_argument('-results_table', type=int, required=False, default=1, help='enable evolution results table visualization')
PARSER.add_argument('-model_distribution', type=int, required=False, default=1, help='enable evolution model distribution visualization')
PARSER.add_argument('-model_evolution', type=int, required=False, default=1, help='enable model evolution visualization')
PARSER.add_argument('-param_distribution', type=int, required=False, default=0, help='enable evolution param distribution visualization')
PARSER.add_argument('-train_time_distribution', type=int, required=False, default=1, help='enable evolution training time distribution visualization')
PARSER.add_argument('-breeding_map', type=int, required=False, default=0, help='enable evolution breeding map visualization')
PARSER.add_argument('-breeding_graph', type=int, required=False, default=0, help='enable evolution breeding graph visualization')
PARSER.add_argument('-fitness_distribution', type=int, required=False, default=1, help='enable evolution fitness distribution visualization')
PARSER.add_argument('-fitness_evolution', type=int, required=False, default=1, help='enable evolution fitness evolution visualization')
PARSER.add_argument('-fitness_dimensions', type=int, required=False, default=1, help='enable evolution fitness dimension visualization')
PARSER.add_argument('-per_iteration', type=int, required=False, default=1, help='enable evolution per iteration visualization')
PARSER.add_argument('-output_file_path_evolve', type=str, required=False, default=None, help='file path of the proportion of valid features')
PARSER.add_argument('-output_file_path_stopping_reason', type=str, required=False, default=None, help='complete customized file path of the data health check output')
PARSER.add_argument('-output_file_path_generator_instructions', type=str, required=False, default=None, help='complete customized file path of the missing data output')
PARSER.add_argument('-output_file_path_current_iteration_meta_data', type=str, required=False, default=None, help='file path of invariant features')
PARSER.add_argument('-output_file_path_iteration_history', type=str, required=False, default=None, help='file path of duplicated features')
PARSER.add_argument('-output_file_path_evolution_history', type=str, required=False, default=None, help='file path of valid features')
PARSER.add_argument('-output_file_path_evolution_gradient', type=str, required=False, default=None, help='file path of the proportion of valid features')
PARSER.add_argument('-s3_output_file_path_modeling', type=str, required=True, default=None, help='S3 file path of the modeling steps output')
PARSER.add_argument('-s3_output_file_path_visualization', type=str, required=True, default=None, help='S3 file path of the visualization output')
ARGS = PARSER.parse_args()


def evolutionary_algorithm(s3_metadata_file_path: str,
                           target: str,
                           features: List[str],
                           models: List[str],
                           train_data_file_path: str,
                           test_data_file_path: str,
                           output_file_path_evolve: str,
                           output_file_path_stopping_reason: str,
                           output_file_path_generator_instructions: str,
                           output_file_path_current_iteration_meta_data: str,
                           output_file_path_iteration_history: str,
                           output_file_path_evolution_history: str,
                           output_file_path_evolution_gradient: str,
                           s3_output_file_path_modeling: str,
                           s3_output_file_path_visualization: str,
                           val_data_file_path: str = None,
                           algorithm: str = 'ga',
                           max_iterations: int = 10,
                           pop_size: int = 64,
                           burn_in_iterations: int = -1,
                           warm_start: bool = True,
                           change_rate: float = 0.1,
                           change_prob: float = 0.85,
                           parents_ratio: float = 0.5,
                           early_stopping: int = 0,
                           convergence: bool = False,
                           convergence_measure: str = 'min',
                           timer_in_seconds: int = 43200,
                           re_populate: bool = False,
                           re_populate_threshold: float = 3.0,
                           max_trials: int = 2,
                           environment_reaction: dict = None,
                           results_table: bool = True,
                           model_distribution: bool = True,
                           model_evolution: bool = True,
                           param_distribution: bool = False,
                           train_time_distribution: bool = True,
                           breeding_map: bool = False,
                           breeding_graph: bool = False,
                           fitness_distribution: bool = True,
                           fitness_evolution: bool = True,
                           fitness_dimensions: bool = True,
                           per_iteration: bool = True
                           ) -> NamedTuple('outputs', [('evolve', str),
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

    :param s3_metadata_file_path:
    :param target:
    :param features:
    :param models:
    :param train_data_file_path:
    :param test_data_file_path:
    :param output_file_path_evolve:
    :param output_file_path_stopping_reason:
    :param output_file_path_generator_instructions:
    :param output_file_path_current_iteration_meta_data:
    :param output_file_path_iteration_history:
    :param output_file_path_evolution_history:
    :param output_file_path_evolution_gradient:
    :param s3_output_file_path_modeling: str
        Path of the output files of the following modeling steps

    :param s3_output_file_path_visualization: str
        Path of the output files of the following visualization step

    :param val_data_file_path:
    :param algorithm:
    :param max_iterations:
    :param pop_size:
    :param burn_in_iterations:
    :param warm_start:
    :param change_rate:
    :param change_prob:
    :param parents_ratio:
    :param early_stopping:
    :param convergence:
    :param convergence_measure:
    :param timer_in_seconds:
    :param re_populate:
    :param re_populate_threshold: float
    :param max_trials:
    :param environment_reaction: dict
    :return: NamedTuple

    """
    if file_exists(file_path=s3_metadata_file_path):
        _metadata: dict = load_file_from_s3(file_path=s3_metadata_file_path)
        _evolutionary_algorithm: EvolutionaryAlgorithm = EvolutionaryAlgorithm(metadata=_metadata)
        if environment_reaction is not None:
            _evolutionary_algorithm.gather_metadata(environment_reaction=environment_reaction)
        _re_populate: bool = _evolutionary_algorithm.check_for_re_population()
        if _re_populate:
            _evolutionary_algorithm.generate_metadata_template(algorithm=algorithm,
                                                               max_iterations=max_iterations,
                                                               pop_size=pop_size,
                                                               burn_in_iterations=burn_in_iterations,
                                                               warm_start=warm_start,
                                                               change_rate=change_rate,
                                                               change_prob=change_prob,
                                                               parents_ratio=parents_ratio,
                                                               early_stopping=early_stopping,
                                                               convergence=convergence,
                                                               convergence_measure=convergence_measure,
                                                               timer_in_seconds=timer_in_seconds,
                                                               target=target,
                                                               features=features,
                                                               models=models,
                                                               train_data_file_path=train_data_file_path,
                                                               test_data_file_path=test_data_file_path,
                                                               val_data_file_path=val_data_file_path,
                                                               re_populate=re_populate,
                                                               max_trials=max_trials,
                                                               re_populate_threshold=re_populate_threshold
                                                               )
            _evolve: str = 'false'
            _stopping_reason: str = None
            _generator_instructions: List[dict] = _evolutionary_algorithm.populate()
        else:
            _check_for_stopping: dict = _evolutionary_algorithm.check_for_stopping()
            _stopping_reason: str = _check_for_stopping.get('stopping_reason')
            if _check_for_stopping.get('evolve'):
                _evolve: str = 'true'
                _generator_instructions: List[dict] = _evolutionary_algorithm.main()
            else:
                _evolve: str = 'false'
                _generator_instructions: List[dict] = []
                _evolutionary_algorithm.generate_visualization_config(path=s3_output_file_path_visualization,
                                                                      results_table=results_table,
                                                                      model_distribution=model_distribution,
                                                                      model_evolution=model_evolution,
                                                                      param_distribution=param_distribution,
                                                                      train_time_distribution=train_time_distribution,
                                                                      breeding_map=breeding_map,
                                                                      breeding_graph=breeding_graph,
                                                                      fitness_distribution=fitness_distribution,
                                                                      fitness_evolution=fitness_evolution,
                                                                      fitness_dimensions=fitness_dimensions,
                                                                      per_iteration=per_iteration
                                                                      )
    else:
        _evolutionary_algorithm: EvolutionaryAlgorithm = EvolutionaryAlgorithm(metadata=None)
        _evolutionary_algorithm.generate_metadata_template(algorithm=algorithm,
                                                           max_iterations=max_iterations,
                                                           pop_size=pop_size,
                                                           burn_in_iterations=burn_in_iterations,
                                                           warm_start=warm_start,
                                                           change_rate=change_rate,
                                                           change_prob=change_prob,
                                                           parents_ratio=parents_ratio,
                                                           early_stopping=early_stopping,
                                                           convergence=convergence,
                                                           convergence_measure=convergence_measure,
                                                           timer_in_seconds=timer_in_seconds,
                                                           target=target,
                                                           features=features,
                                                           models=models,
                                                           train_data_file_path=train_data_file_path,
                                                           test_data_file_path=test_data_file_path,
                                                           val_data_file_path=val_data_file_path,
                                                           re_populate=re_populate,
                                                           max_trials=max_trials,
                                                           re_populate_threshold=re_populate_threshold
                                                           )
        _evolve: str = 'false'
        _stopping_reason: str = None
        _generator_instructions: List[dict] = _evolutionary_algorithm.populate()
    _enriched_generator_instructions: List[dict] = []
    if len(_generator_instructions) > 0:
        for individual in _generator_instructions:
            _instructions: dict = copy.deepcopy(individual)
            _instructions.update({'file_path_model_artifact': os.path.join(s3_output_file_path_modeling, f'model_artifact_{_instructions.get("id")}.p'),
                                  'file_path_model_param': os.path.join(s3_output_file_path_modeling, f'model_param{_instructions.get("id")}.json'),
                                  'file_path_model_fitness': os.path.join(s3_output_file_path_modeling, f'model_fitness{_instructions.get("id")}.json'),
                                  })
            _enriched_generator_instructions.append(_instructions)
    for file_path, obj in [(output_file_path_evolve, _evolve),
                           (output_file_path_stopping_reason, _stopping_reason),
                           (output_file_path_generator_instructions, _enriched_generator_instructions),
                           (output_file_path_current_iteration_meta_data, _evolutionary_algorithm.metadata['current_iteration_meta_data']),
                           (output_file_path_iteration_history, _evolutionary_algorithm.metadata['iteration_history']),
                           (output_file_path_evolution_history, _evolutionary_algorithm.metadata['evolution_history']),
                           (output_file_path_evolution_gradient, _evolutionary_algorithm.metadata['evolution_gradient'])
                           ]:
        file_handler(file_path=file_path, obj=obj)
    save_file_to_s3(file_path=s3_metadata_file_path, obj=_evolutionary_algorithm.metadata)
    Log().log(msg=f'Save evolutionary metadata: {s3_metadata_file_path}')
    return [_evolve,
            _stopping_reason,
            _enriched_generator_instructions,
            _evolutionary_algorithm.metadata['current_iteration_meta_data'],
            _evolutionary_algorithm.metadata['iteration_history'],
            _evolutionary_algorithm.metadata['evolution_history'],
            _evolutionary_algorithm.metadata['evolution_gradient']
            ]


if __name__ == '__main__':
    evolutionary_algorithm(s3_metadata_file_path=ARGS.s3_metadata_file_path,
                           target=ARGS.target,
                           features=ARGS.features,
                           models=ARGS.models,
                           train_data_file_path=ARGS.train_data_file_path,
                           test_data_file_path=ARGS.test_data_file_path,
                           output_file_path_evolve=ARGS.output_file_path_evolve,
                           output_file_path_stopping_reason=ARGS.output_file_path_stopping_reason,
                           output_file_path_generator_instructions=ARGS.output_file_path_generator_instructions,
                           output_file_path_current_iteration_meta_data=ARGS.output_file_path_current_iteration_meta_data,
                           output_file_path_iteration_history=ARGS.output_file_path_iteration_history,
                           output_file_path_evolution_history=ARGS.output_file_path_evolution_history,
                           output_file_path_evolution_gradient=ARGS.output_file_path_evolution_gradient,
                           s3_output_file_path_modeling=ARGS.s3_output_file_path_modeling,
                           s3_output_file_path_visualization=ARGS.s3_output_file_path_visualization,
                           val_data_file_path=ARGS.val_data_file_path,
                           algorithm=ARGS.algorithm,
                           max_iterations=ARGS.max_iterations,
                           pop_size=ARGS.pop_size,
                           burn_in_iterations=ARGS.burn_in_iterations,
                           warm_start=bool(ARGS.warm_start),
                           change_rate=ARGS.change_rate,
                           change_prob=ARGS.change_prob,
                           parents_ratio=ARGS.parents_ratio,
                           early_stopping=ARGS.early_stopping,
                           convergence=bool(ARGS.convergence),
                           convergence_measure=ARGS.convergence_measure,
                           timer_in_seconds=ARGS.timer_in_seconds,
                           re_populate=bool(ARGS.re_populate),
                           re_populate_threshold=ARGS.re_populate_threshold,
                           max_trials=ARGS.max_trials,
                           environment_reaction=ARGS.environment_reaction,
                           results_table=bool(ARGS.results_table),
                           model_distribution=bool(ARGS.model_distribution),
                           model_evolution=bool(ARGS.model_evolution),
                           param_distribution=bool(ARGS.param_distribution),
                           train_time_distribution=bool(ARGS.train_time_distribution),
                           breeding_map=bool(ARGS.breeding_map),
                           breeding_graph=bool(ARGS.breeding_graph),
                           fitness_distribution=bool(ARGS.fitness_distribution),
                           fitness_evolution=bool(ARGS.fitness_evolution),
                           fitness_dimensions=bool(ARGS.fitness_dimensions),
                           per_iteration=bool(ARGS.per_iteration)
                           )
