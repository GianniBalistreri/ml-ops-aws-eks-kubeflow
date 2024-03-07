"""

Task: ... (Function to run in container)

"""

import argparse
import ast
import copy
import os

from aws import file_exists, load_file_from_s3, save_file_to_s3
from custom_logger import Log
from evolutionary_algorithm import EvolutionaryAlgorithm
from file_handler import file_handler
from resource_metrics import get_available_cpu, get_cpu_utilization, get_cpu_utilization_per_core, get_memory, get_memory_utilization
from typing import List, NamedTuple


PARSER = argparse.ArgumentParser(description="evolutionary algorithm")
PARSER.add_argument('-s3_metadata_file_path', type=str, required=True, default=None, help='S3 file path of the evolutionary algorithm metadata')
PARSER.add_argument('-target', type=str, required=True, default=None, help='name of the target feature')
PARSER.add_argument('-features', nargs='+', required=True, default=None, help='name of the features')
PARSER.add_argument('-models', nargs='+', required=True, default=None, help='name of the machine learning models')
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
PARSER.add_argument('-crossover', type=int, required=False, default=1, help='whether to apply crossover inheritance strategy or not (genetic algorithm only)')
PARSER.add_argument('-early_stopping', type=int, required=False, default=0, help='whether to stop evolution early if conditions are met or not')
PARSER.add_argument('-convergence', type=int, required=False, default=0, help='whether to check convergence of the population or not')
PARSER.add_argument('-convergence_measure', type=str, required=False, default='min', help='measurement for convergence')
PARSER.add_argument('-stagnation', type=int, required=False, default=0, help='whether to check stagnation of the population or not')
PARSER.add_argument('-stagnation_threshold', type=float, required=False, default=98.0, help='stagnation threshold')
PARSER.add_argument('-timer_in_seconds', type=int, required=False, default=43200, help='maximum number of seconds for evolution')
PARSER.add_argument('-re_populate', type=int, required=False, default=1, help='whether to re-populate initial population because of poor individual fitness score')
PARSER.add_argument('-re_populate_threshold', type=float, required=False, default=3.0, help='fitness score threshold for re-population')
PARSER.add_argument('-max_trials', type=int, required=False, default=2, help='number of trials for re-population before continuing evolution process')
PARSER.add_argument('-output_file_path_evolve', type=str, required=True, default=None, help='file path of how to proceed with the evolutionary algorithm output')
PARSER.add_argument('-output_file_path_stopping_reason', type=str, required=True, default=None, help='file path of the stopping reason output')
PARSER.add_argument('-output_file_path_individual_idx', type=str, required=True, default=None, help='file path of the individual index of instruction list to proceed output')
PARSER.add_argument('-s3_output_file_path_generator_instructions', type=str, required=True, default=None, help='S3 file path of the generator instruction output')
PARSER.add_argument('-s3_output_file_path_modeling', type=str, required=True, default=None, help='S3 file path of the modeling steps output')
ARGS = PARSER.parse_args()


def evolutionary_algorithm(s3_metadata_file_path: str,
                           target: str,
                           features: List[str],
                           models: List[str],
                           train_data_file_path: str,
                           test_data_file_path: str,
                           output_file_path_evolve: str,
                           output_file_path_stopping_reason: str,
                           output_file_path_individual_idx: str,
                           s3_output_file_path_generator_instructions: str,
                           s3_output_file_path_modeling: str,
                           val_data_file_path: str = None,
                           algorithm: str = 'ga',
                           max_iterations: int = 10,
                           pop_size: int = 64,
                           burn_in_iterations: int = -1,
                           warm_start: bool = True,
                           change_rate: float = 0.1,
                           change_prob: float = 0.85,
                           parents_ratio: float = 0.5,
                           crossover: bool = True,
                           early_stopping: bool = False,
                           convergence: bool = False,
                           convergence_measure: str = 'min',
                           stagnation: bool = False,
                           stagnation_threshold: float = 98.0,
                           timer_in_seconds: int = 43200,
                           re_populate: bool = False,
                           re_populate_threshold: float = 3.0,
                           max_trials: int = 2
                           ) -> NamedTuple('outputs', [('evolve', int),
                                                       ('stopping_reason', str),
                                                       ('idx', list)
                                                       ]
                                           ):
    """
    Optimize machine learning models

    :param s3_metadata_file_path: str
        Complete file path of the metadata

    :param target: str
        Name of the target feature

    :param features: List[str]
        Name of the features

    :param models: List[str]
        Abbreviated name of the machine learning models

    :param train_data_file_path: str
        Complete file path of the training data

    :param test_data_file_path: str
        Complete file path of the test data

    :param output_file_path_evolve: str
        File path of the evolution status output

    :param output_file_path_stopping_reason: str
        File path of the stopping reason output

    :param output_file_path_individual_idx: str
        File path of the individual index of the instruction list to proceed output

    :param s3_output_file_path_generator_instructions: str
        Path of the generator instruction output for the following modeling steps

    :param s3_output_file_path_modeling: str
        Path of the output files of the following modeling steps

    :param val_data_file_path: str
        Complete file path of the validation data set

    :param algorithm: str
        Abbreviated name of the evolutionary algorithm
            -> ga: Genetic Algorithm
            -> si: Swarm Intelligence (POS)
            -> ga_si: Alternating Genetic Algorithm and Swarm Intelligence (POS)
            -> si_ga: Alternating Swarm Intelligence (POS) and Genetic Algorithm
            -> random: Choose either Genetic Algorithm or Swarm Intelligence per iteration randomly

    :param max_iterations: int
        Maximum number of iterations

    :param pop_size: int
        Size of the population

    :param burn_in_iterations: int
        Number of burn-in iterations

    :param warm_start: bool
        Whether to run with warm start (one individual has standard hyperparameter settings)

    :param change_rate: float
        Rate of the hyperparameter change (mutation / adjustment)

    :param change_prob: float
        Probability of changing hyperparameter (mutation / adjustment)

    :param parents_ratio: float
        Ratio of parenthood

    :param crossover: bool
        Whether to apply crossover inheritance strategy or not (generic algorithm only)

    :param early_stopping: bool
        Whether to enable early stopping or not

    :param convergence: bool
        Whether to enable convergence

    :param convergence_measure: str
        Abbreviated name of the convergence measurement

    :param stagnation: bool
        Whether to enable gradient stagnation or not

    :param stagnation_threshold: float
        Threshold to identify stagnation

    :param timer_in_seconds: int
        Timer in seconds for stopping evolution

    :param re_populate: bool
        Whether to re-populate because of poor performance of the entire population or not

    :param re_populate_threshold: float
        Threshold to decide to re-populate

    :param max_trials: int
        Maximum number of trials for re-population

    :return: NamedTuple
        Whether to continue evolution or not and stopping reason if not continuing and individual index of instruction list
    """
    _cpu_available: int = get_available_cpu(logging=True)
    _memory_total: float = get_memory(total=True, logging=True)
    _memory_available: float = get_memory(total=False, logging=True)
    if file_exists(file_path=s3_metadata_file_path):
        _metadata: dict = load_file_from_s3(file_path=s3_metadata_file_path)
        Log().log(msg=f'Load metadata file: {s3_metadata_file_path}')
        _evolutionary_algorithm: EvolutionaryAlgorithm = EvolutionaryAlgorithm(metadata=_metadata)
        if _evolutionary_algorithm.metadata.get('continue_evolution'):
            Log().log(msg=f'Continue previous evolution')
            _evolve: bool = True
            _stopping_reason: str = None
            _generator_instructions: List[dict] = _evolutionary_algorithm.main()
        else:
            _re_populate: bool = _evolutionary_algorithm.check_for_re_population()
            if _re_populate:
                Log().log(msg=f'Re-populate')
                _evolutionary_algorithm.generate_metadata_template(algorithm=algorithm,
                                                                   max_iterations=max_iterations,
                                                                   pop_size=pop_size,
                                                                   burn_in_iterations=burn_in_iterations,
                                                                   warm_start=warm_start,
                                                                   change_rate=change_rate,
                                                                   change_prob=change_prob,
                                                                   parents_ratio=parents_ratio,
                                                                   crossover=crossover,
                                                                   early_stopping=early_stopping,
                                                                   convergence=convergence,
                                                                   convergence_measure=convergence_measure,
                                                                   stagnation=stagnation,
                                                                   stagnation_threshold=stagnation_threshold,
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
                _evolve: bool = True
                _stopping_reason: str = None
                _generator_instructions: List[dict] = _evolutionary_algorithm.populate()
            else:
                _check_for_stopping: dict = _evolutionary_algorithm.check_for_stopping()
                _stopping_reason: str = _check_for_stopping.get('stopping_reason')
                _evolve: bool = _check_for_stopping.get('evolve')
                _generator_instructions: List[dict] = _evolutionary_algorithm.main()
    else:
        Log().log(msg='Initialize population')
        _evolutionary_algorithm: EvolutionaryAlgorithm = EvolutionaryAlgorithm(metadata=None)
        _evolutionary_algorithm.generate_metadata_template(algorithm=algorithm,
                                                           max_iterations=max_iterations,
                                                           pop_size=pop_size,
                                                           burn_in_iterations=burn_in_iterations,
                                                           warm_start=warm_start,
                                                           change_rate=change_rate,
                                                           change_prob=change_prob,
                                                           parents_ratio=parents_ratio,
                                                           crossover=crossover,
                                                           early_stopping=early_stopping,
                                                           convergence=convergence,
                                                           convergence_measure=convergence_measure,
                                                           stagnation=stagnation,
                                                           stagnation_threshold=stagnation_threshold,
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
        _evolve: bool = True
        _stopping_reason: str = None
        _generator_instructions: List[dict] = _evolutionary_algorithm.populate()
    _enriched_generator_instructions: List[dict] = []
    _idx: List[int] = []
    if len(_generator_instructions) > 0:
        for individual in _generator_instructions:
            _instructions: dict = copy.deepcopy(individual)
            if individual.get('idx') is not None:
                _idx.append(individual.get('idx'))
                _instructions.update({'model_artifact_path': os.path.join(s3_output_file_path_modeling, f'model_artifact_{_instructions.get("id")}.joblib'),
                                      'model_param_path': os.path.join(s3_output_file_path_modeling, f'model_param_{_instructions.get("id")}.json'),
                                      'model_metadata_path': os.path.join(s3_output_file_path_modeling, f'model_metadata_{_instructions.get("id")}.json'),
                                      'model_fitness_path': os.path.join(s3_output_file_path_modeling, f'model_fitness_{_instructions.get("id")}.json'),
                                      'evaluate_train_data_path': os.path.join(s3_output_file_path_modeling, f'evaluate_train_data_{_instructions.get("id")}.json'),
                                      'evaluate_test_data_path': os.path.join(s3_output_file_path_modeling, f'evaluate_test_data_{_instructions.get("id")}.json'),
                                      'evaluate_val_data_path': os.path.join(s3_output_file_path_modeling, f'evaluate_val_data_{_instructions.get("id")}.json'),
                                      })
                if _evolutionary_algorithm.metadata['current_iteration'] > 0:
                    _input_param_file_path: str = os.path.join(s3_output_file_path_modeling, f'model_input_param_{_instructions.get("id")}.json')
                    save_file_to_s3(file_path=_input_param_file_path, obj=individual.get('params'))
                    Log().log(msg=f'Save evolved hyperparameter: {_input_param_file_path}')
                    _instructions.update({'model_input_param_path': _input_param_file_path})
            _enriched_generator_instructions.append(_instructions)
        save_file_to_s3(file_path=s3_output_file_path_generator_instructions, obj=_enriched_generator_instructions)
        Log().log(msg=f'Save generator instructions: {s3_output_file_path_generator_instructions}')
    file_handler(file_path=output_file_path_evolve, obj=int(_evolve))
    Log().log(msg=f'Continue evolution: {_evolve}')
    file_handler(file_path=output_file_path_stopping_reason, obj=_stopping_reason)
    file_handler(file_path=output_file_path_individual_idx, obj=_idx)
    if not _evolve:
        Log().log(msg=f'Stopping reason: {_stopping_reason}')
    #save_file_to_s3(file_path=s3_metadata_file_path, obj=_evolutionary_algorithm.metadata)
    #Log().log(msg=f'Save evolutionary metadata: {s3_metadata_file_path}')
    _s3_metadata_file_path_temp: str = f'{s3_metadata_file_path.split(".")[0]}__temp__.{s3_metadata_file_path.split(".")[-1]}'
    save_file_to_s3(file_path=_s3_metadata_file_path_temp, obj=_evolutionary_algorithm.metadata)
    Log().log(msg=f'Save temporary evolutionary metadata: {_s3_metadata_file_path_temp}')
    _cpu_utilization: float = get_cpu_utilization(interval=1, logging=True)
    _cpu_utilization_per_cpu: List[float] = get_cpu_utilization_per_core(interval=1, logging=True)
    _memory_utilization: float = get_memory_utilization(logging=True)
    _memory_available = get_memory(total=False, logging=True)
    return [_evolve, _stopping_reason, _idx]


if __name__ == '__main__':
    if ARGS.features:
        ARGS.features = ast.literal_eval(ARGS.features[0])
    if ARGS.models:
        ARGS.models = ast.literal_eval(ARGS.models[0])
    evolutionary_algorithm(s3_metadata_file_path=ARGS.s3_metadata_file_path,
                           target=ARGS.target,
                           features=ARGS.features,
                           models=ARGS.models,
                           train_data_file_path=ARGS.train_data_file_path,
                           test_data_file_path=ARGS.test_data_file_path,
                           output_file_path_evolve=ARGS.output_file_path_evolve,
                           output_file_path_stopping_reason=ARGS.output_file_path_stopping_reason,
                           output_file_path_individual_idx=ARGS.output_file_path_individual_idx,
                           s3_output_file_path_generator_instructions=ARGS.s3_output_file_path_generator_instructions,
                           s3_output_file_path_modeling=ARGS.s3_output_file_path_modeling,
                           val_data_file_path=ARGS.val_data_file_path,
                           algorithm=ARGS.algorithm,
                           max_iterations=ARGS.max_iterations,
                           pop_size=ARGS.pop_size,
                           burn_in_iterations=ARGS.burn_in_iterations,
                           warm_start=bool(ARGS.warm_start),
                           change_rate=ARGS.change_rate,
                           change_prob=ARGS.change_prob,
                           parents_ratio=ARGS.parents_ratio,
                           crossover=bool(ARGS.crossover),
                           early_stopping=bool(ARGS.early_stopping),
                           convergence=bool(ARGS.convergence),
                           convergence_measure=ARGS.convergence_measure,
                           stagnation=bool(ARGS.stagnation),
                           stagnation_threshold=ARGS.stagnation_threshold,
                           timer_in_seconds=ARGS.timer_in_seconds,
                           re_populate=bool(ARGS.re_populate),
                           re_populate_threshold=ARGS.re_populate_threshold,
                           max_trials=ARGS.max_trials
                           )
