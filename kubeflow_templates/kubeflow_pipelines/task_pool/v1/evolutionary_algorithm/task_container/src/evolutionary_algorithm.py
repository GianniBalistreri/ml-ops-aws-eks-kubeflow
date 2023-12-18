"""

Evolutionary algorithm

"""

import copy
import numpy as np
import random
import pandas as pd

from custom_logger import Log
from datetime import datetime
from typing import List


EVOLUTIONARY_ALGORITHMS: List[str] = ['gi', 'si']


class EvolutionaryAlgorithmException(Exception):
    """
    Class for handling exception for class EvolutionaryAlgorithm
    """
    pass


class EvolutionaryAlgorithm:
    """
    Class for applying evolutionary algorithm for machine learning optimization
    """
    def __init__(self, metadata: dict = None):
        """
        :param metadata: dict
            Metadata collection
        """
        self.metadata: dict = metadata
        self.parent_child_pairs: List[tuple] = []
        if self.metadata is not None:
            if len(self.metadata['end_time']) > 0 and len(self.metadata['start_time']) == len(self.metadata['end_time']):
                _current_time: datetime = datetime.now()
                if self.metadata['current_iteration'] > 0 and _current_time > self.metadata['end_time'][-1]:
                    self.metadata['start_time'].append(_current_time)

    def _adjust_swarm(self) -> List[dict]:
        """
        Adjust swarm based on POS method

        :return: dict
        """
        _adjustments: List[dict] = []
        self._select_best_individual()
        _new_id: int = np.array(self.metadata['generated_individuals']).argmax()
        for idx in range(0, self.metadata['pop_size'], 1):
            if idx != self.metadata['best_global_idx'] and idx != self.metadata['best_local_idx']:
                _new_id += 1
                self.metadata['generated_individuals'].append(_new_id)
                if np.random.uniform(low=0, high=1) > self.metadata['change_prob']:
                    _adjustment_instructions: dict = dict(idx=idx,
                                                          id=_new_id,
                                                          parent=self.metadata['best_global_idx'],
                                                          model_name=random.choice([self.metadata['models']]),
                                                          params=None,
                                                          param_rate=1.0,
                                                          warm_start=0
                                                          )
                    _adjustments.append(_adjustment_instructions)
                    Log().log(msg=f'Define new model for individual {idx}')
                else:
                    _param: dict = self._pso(idx=idx)
                    if np.random.uniform(low=0, high=1) > self.metadata['change_prob']:
                        _param_rate: float = self.metadata['change_rate']
                    else:
                        _param_rate: float = 0.0
                    _adjustment_instructions: dict = dict(idx=idx,
                                                          id=_new_id,
                                                          parent=self.metadata['best_global_idx'],
                                                          model_name=self.metadata['current_iteration_meta_data']['model_name'][idx],
                                                          params=self.metadata['current_iteration_meta_data']['param'][idx],
                                                          param_rate=_param_rate,
                                                          warm_start=0
                                                          )
                    _adjustments.append(_adjustment_instructions)
                    #Log().log(msg=f'Adjust individual {idx}')
        return _adjustments

    def _crossover(self, parent: int, child: int) -> dict:
        """
        Mutate individuals after mixing the individual genes using cross-over method

        :param parent: int
            Index number of parent in population

        :param child: int
            Index number of child in population
        """
        _new_genes: dict = copy.deepcopy(self.metadata['current_iteration_meta_data']['param'][child])
        _inherit_genes: List[str] = list(self.metadata['current_iteration_meta_data']['param'][parent].keys())
        _x: int = 0
        for _ in range(0, round(len(_inherit_genes) * self.metadata['parents_ratio']), 1):
            while True:
                _gene: str = np.random.choice(_inherit_genes)
                if self.metadata['current_iteration_meta_data']['param'][child][_gene] != self.metadata['current_iteration_meta_data']['param'][parent][_gene] or _x == 20:
                    _x = 0
                    _new_genes[_gene] = copy.deepcopy(self.metadata['current_iteration_meta_data']['param'][parent][_gene])
                    break
                else:
                    _x += 1
        Log().log(msg=f'Cross-over individual {child} with individual {parent}')
        return _new_genes

    def _inherit(self) -> None:
        """
        Inheritance from one parent to one child
        """
        _min_matching_length: int = len(self.metadata['parents_idx']) if len(self.metadata['parents_idx']) <= len(self.metadata['children_idx']) else len(self.metadata['children_idx'])
        for c in range(0, _min_matching_length, 1):
            self.parent_child_pairs.append(tuple([self.metadata['parents_idx'][c], self.metadata['children_idx'][c]]))

    def _is_gradient_converged(self, threshold: float = 0.05) -> bool:
        """
        Check whether evolutionary gradient has converged into optimum

        :param threshold: float
            Conversion threshold of relative difference between maximum fitness score and comparison fitness score

        :return bool
            Whether to stop evolution because the hole iteration achieve very similar gradient score or not
        """
        _threshold: float = threshold if threshold > 0 else 0.05
        _threshold_score: float = self.metadata['evolution_gradient'].get('max')[-1] - (self.metadata['evolution_gradient'].get('max')[-1] * _threshold)
        if self.metadata['convergence_measure'] == 'median':
            if self.metadata['evolution_gradient'].get('median')[-1] >= _threshold_score:
                return True
            else:
                return False
        elif self.metadata['convergence_measure'] == 'mean':
            if self.metadata['evolution_gradient'].get('mean')[-1] >= _threshold_score:
                return True
            else:
                return False
        else:
            if self.metadata['evolution_gradient'].get('min')[-1] >= _threshold_score:
                return True
            else:
                return False

    def _is_gradient_stagnating(self,
                                min_fitness: bool = True,
                                median_fitness: bool = True,
                                mean_fitness: bool = True,
                                max_fitness: bool = True
                                ) -> bool:
        """
        Check whether evolutionary gradient (best fitness metric of iteration) has not increased a certain amount of iterations

        :param min_fitness: bool
            Use minimum fitness score each iteration to evaluate stagnation

        :param median_fitness: bool
            Use median fitness score each iteration to evaluate stagnation

        :param mean_fitness: bool
            Use mean fitness score each iteration to evaluate stagnation

        :param max_fitness: bool
            Use maximum fitness score each iteration to evaluate stagnation

        :return bool
            Whether to stop evolution early because of the stagnation of gradient or not
        """
        _gradients: int = 0
        _stagnating: int = 0
        if min_fitness:
            _gradients += 1
            _stagnating = int(len(self.metadata['evolution_gradient'].get('min')) - np.array(self.metadata['evolution_gradient'].get('min')).argmax() >= self.metadata['early_stopping'])
        if median_fitness:
            _gradients += 1
            _stagnating = int(len(self.metadata['evolution_gradient'].get('median')) - np.array(self.metadata['evolution_gradient'].get('median')).argmax() >= self.metadata['early_stopping'])
        if mean_fitness:
            _gradients += 1
            _stagnating = int(len(self.metadata['evolution_gradient'].get('mean')) - np.array(self.metadata['evolution_gradient'].get('mean')).argmax() >= self.metadata['early_stopping'])
        if max_fitness:
            _gradients += 1
            _stagnating = int(len(self.metadata['evolution_gradient'].get('max')) - np.array(self.metadata['evolution_gradient'].get('max')).argmax() >= self.metadata['early_stopping'])
        if _gradients == _stagnating:
            return True
        else:
            return False

    def _natural_selection(self) -> None:
        """
        Select best individuals of population as parents for next iteration
        """
        # Calculate number of parents within current iteration:
        _count_parents: int = int(self.metadata['pop_size'] * self.metadata['parents_ratio'])
        # Rank individuals according to their fitness score:
        _sorted_fitness_matrix: pd.DataFrame = pd.DataFrame(data=dict(fitness=self.metadata['current_iteration_meta_data'].get('fitness_score'))).sort_values(by='fitness', axis=0, ascending=False)
        # Set parents:
        self.metadata['parents_idx'] = _sorted_fitness_matrix[0:_count_parents].index.values.tolist()
        # Set children:
        self.metadata['children_idx'] = _sorted_fitness_matrix[_count_parents:].index.values.tolist()

    def _mating_pool(self) -> List[dict]:
        """
        Mutate genes of chosen parents

        :return: List[dict]
        """
        _mutations: List[dict] = []
        self._natural_selection()
        self._inherit()
        _new_id: int = np.array(self.metadata['generated_individuals']).argmax()
        for parent, child in self.parent_child_pairs:
            _new_id += 1
            self.metadata['generated_individuals'].append(_new_id)
            if np.random.uniform(low=0, high=1) > self.metadata['change_prob']:
                _mutation_instructions: dict = dict(idx=child,
                                                    id=_new_id,
                                                    parent=parent,
                                                    model_name=random.choice([self.metadata['models']]),
                                                    params=None,
                                                    param_rate=1.0,
                                                    warm_start=0
                                                    )
                _mutations.append(_mutation_instructions)
                Log().log(msg=f'Define new model for individual {child}')
            else:
                if self.metadata['crossover']:
                    if self.metadata['current_iteration_meta_data']['model_name'][parent] == self.metadata['current_iteration_meta_data']['model_name'][child]:
                        _params: dict = self._crossover(parent=parent, child=child)
                    else:
                        _params: dict = self.metadata['current_iteration_meta_data']['param'][parent]
                else:
                    _params: dict = self.metadata['current_iteration_meta_data']['param'][parent]
                _mutation_instructions: dict = dict(idx=child,
                                                    id=_new_id,
                                                    parent=parent,
                                                    model_name=self.metadata['current_iteration_meta_data']['model_name'][parent],
                                                    params=_params,
                                                    param_rate=self.metadata['change_rate'],
                                                    warm_start=0
                                                    )
                _mutations.append(_mutation_instructions)
                Log().log(msg=f'Mutate individual {child}')
        return _mutations

    def _pso(self, idx: int) -> dict:
        """
        Apply particle swarm optimization algorithm

        :param idx: int
            Index of the individual within the population

        :return: dict
        """
        _new_characteristics: dict = copy.deepcopy(self.metadata['current_iteration_meta_data']['param'][idx])
        _inherit_genes: List[str] = list(self.metadata['current_iteration_meta_data']['param'][idx].keys())
        _x: int = 0
        for _ in range(0, round(len(_inherit_genes) * self.metadata['change_rate']), 1):
            while True:
                _gene: str = np.random.choice(_inherit_genes)
                if self.metadata['current_iteration_meta_data']['param'][idx][_gene] != self.metadata['current_iteration_meta_data']['param'][self.metadata['best_global_idx'][-1]][_gene] or _x == 20:
                    _x = 0
                    _new_characteristics[_gene] = copy.deepcopy(self.metadata['current_iteration_meta_data']['param'][self.metadata['best_global_idx'][-1]][_gene])
                    break
                else:
                    _x += 1
        Log().log(msg=f'Adjust individual {idx} based on individual {self.metadata["best_global_idx"][-1]}')
        return _new_characteristics

    def _select_best_individual(self) -> None:
        """
        Select current best global and local individual
        """
        _best_global_idx: int = np.array(self.metadata['current_iteration_meta_data']['fitness_score']).argmax()
        _other_idx: List[float] = copy.deepcopy(self.metadata['current_iteration_meta_data']['fitness_score'])
        del _other_idx[_best_global_idx]
        _best_local_idx: int = np.array(_other_idx).argmax()
        if _best_global_idx <= _best_local_idx:
            _best_local_idx += 1
        self.metadata['best_global_idx'] = _best_global_idx
        self.metadata['best_local_idx'] = _best_local_idx
        Log().log(msg=f'Best local individual {_best_local_idx}')
        Log().log(msg=f'Best global individual {_best_global_idx}')

    def _set_iteration_algorithm(self) -> None:
        """
        Set evolutionary algorithm for current iteration
        """
        if self.metadata['algorithm'] == 'random':
            _algorithm: str = random.choice(EVOLUTIONARY_ALGORITHMS)
        elif self.metadata['algorithm'] in ['ga_si', 'si_ga']:
            if self.metadata['current_iteration_algorithm'][-1] == 'ga':
                _algorithm: str = 'si'
            else:
                _algorithm: str = 'ga'
        else:
            _algorithm: str = self.metadata['algorithm']
        self.metadata['current_iteration_algorithm'].append(_algorithm)

    def check_for_re_population(self) -> bool:
        """
        Check whether to re-populate initial environment because of poor fitness of all individuals

        :return: bool
            Whether to re-populate environment or not
        """
        if self.metadata['current_iteration'] == 0 and self.metadata['n_trial'] <= self.metadata['max_trials']:
            _best_score: float = np.array(self.metadata['current_iteration_meta_data']['fitness_score']).argmax()
            if _best_score < self.metadata['re_populate_threshold']:
                self.metadata['n_trial'] += 1
                Log().log(msg=f'Re-populate environment because of poor scoring (Trial: {self.metadata["n_trial"]})')
                return True
        return False

    def check_for_stopping(self) -> dict:
        """
        Check whether to stop algorithm or not

        :return: dict
            Whether to stop algorithm or not
        """
        _results: dict = dict(evolve=True, stopping_reason=None)
        if self.metadata['current_iteration_meta_data']['iteration'] > self.metadata['burn_in_iterations']:
            if self.metadata['convergence']:
                if self._is_gradient_converged(threshold=0.05):
                    _results['evolve'] = False
                    _results['stopping_reason'] = 'gradient_converged'
                    Log().log(msg=f'Fitness metric (gradient) has converged. Therefore the evolution stops at iteration {self.metadata["current_iteration_meta_data"]["iteration"]}')
            if self.metadata['early_stopping'] > 0:
                if self._is_gradient_stagnating(min_fitness=True, median_fitness=True, mean_fitness=True, max_fitness=True):
                    _results['evolve'] = False
                    _results['stopping_reason'] = 'gradient_stagnating'
                    Log().log(msg=f'Fitness metric (gradient) per iteration has not increased a certain amount of iterations ({self.metadata["early_stopping"]}). Therefore the evolution stops early at iteration {self.metadata["current_iteration_meta_data"]["iteration"]}')
        _current_runtime_in_seconds: int = (datetime.now() - self.metadata['start_time'][-1]).seconds
        if _current_runtime_in_seconds >= self.metadata['timer']:
            _results['evolve'] = False
            _results['stopping_reason'] = 'time_exceeded'
            Log().log(msg=f'Time exceeded:{self.metadata["timer"]} by {_current_runtime_in_seconds} seconds')
        if self.metadata['current_iteration_meta_data']['iteration'] > self.metadata['max_iterations']:
            _results['evolve'] = False
            _results['stopping_reason'] = 'max_iteration_evolved'
            Log().log(msg=f'Maximum number of iterations reached: {self.metadata["max_iterations"]}')
        if _results.get('evolve'):
            self.metadata['current_iteration'] += 1
        else:
            self.metadata['stopping_reason'].append(_results['stopping_reason'])
            self.metadata['end_time'].append(datetime.now())
        return _results

    def gather_metadata(self, environment_reaction: dict) -> None:
        """
        Gather observed metadata of current iteration

        :param environment_reaction: dict
        """
        self.metadata['iteration_history']['time'].append((datetime.now() - self.metadata['start_time'][-1]).seconds)
        if self.metadata['current_iteration_algorithm'][-1] == 'ga':
            _fittest_individual_previous_iteration: List[int] = self.metadata['parents_idx']
        else:
            _fittest_individual_previous_iteration: List[int] = [self.metadata['best_global_idx'][-1], self.metadata['best_local_idx'][-1]]
        for i in range(0, self.metadata['pop_size'], 1):
            if i in _fittest_individual_previous_iteration:
                pass
            else:
                # current iteration metadata:
                self.metadata['current_iteration_meta_data']['id'].append(environment_reaction[str(i)]['id'])
                self.metadata['current_iteration_meta_data']['model_name'].append(environment_reaction[str(i)]['model_name'])
                self.metadata['current_iteration_meta_data']['param'].append(environment_reaction[str(i)]['param'])
                self.metadata['current_iteration_meta_data']['param_changed'].append(environment_reaction[str(i)]['param_changed'])
                self.metadata['current_iteration_meta_data']['fitness_metric'].append(environment_reaction[str(i)]['fitness_metric'])
                self.metadata['current_iteration_meta_data']['fitness_score'].append(environment_reaction[str(i)]['fitness_score'])
                Log().log(f'Fitness score {environment_reaction[str(i)]["fitness_score"]} of individual {i}')
                Log().log(f'Fitness metric {environment_reaction[str(i)]["fitness_metric"]} of individual {i}')
                # evolution history:
                self.metadata['evolution_history']['id'].append(environment_reaction[str(i)]['id'])
                self.metadata['evolution_history']['iteration'].append(_metadata['current_iteration'])
                self.metadata['evolution_history']['model_name'].append(environment_reaction[str(i)]['model_name'])
                self.metadata['evolution_history']['parent'].append(environment_reaction[str(i)]['parent'])
                self.metadata['evolution_history']['change_type'].append(environment_reaction[str(i)]['change_type'])
                self.metadata['evolution_history']['fitness_score'].append(environment_reaction[str(i)]['fitness_score'])
                self.metadata['evolution_history']['ml_metric'].append(environment_reaction[str(i)]['fitness_metric'])
                self.metadata['evolution_history']['train_test_diff'].append(environment_reaction[str(i)]['fitness_score'])
                self.metadata['evolution_history']['train_time_in_seconds'].append(environment_reaction[str(i)]['fitness_score'])
                self.metadata['evolution_history']['original_ml_train_metric'].append(environment_reaction[str(i)]['fitness_score'])
                self.metadata['evolution_history']['original_ml_test_metric'].append(environment_reaction[str(i)]['fitness_score'])
                self.evolution_history.get('id').append(copy.deepcopy(self.population[idx].id))
                self.evolution_history.get('generation').append(copy.deepcopy(self.current_generation_meta_data['generation']))
                self.evolution_history.get('model_name').append(copy.deepcopy(self.population[idx].model_name))
                self.evolution_history.get('change_type').append(copy.deepcopy(self.population[idx].model_param_mutation))
                self.generation_history['population']['gen_{}'.format(self.current_generation_meta_data['generation'])]['id'].append(copy.deepcopy(self.population[idx].id))
                self.generation_history['population']['gen_{}'.format(self.current_generation_meta_data['generation'])]['model'].append(copy.deepcopy(self.population[idx].model_name))
        # evolution gradient:
        self.metadata['evolution_gradient']['min'].append(copy.deepcopy(min(self.metadata['current_generation_meta_data'].get('fitness_score'))))
        self.metadata['evolution_gradient']['median'].append(copy.deepcopy(np.median(self.metadata['current_generation_meta_data'].get('fitness_score'))))
        self.metadata['evolution_gradient']['mean'].append(copy.deepcopy(np.mean(self.metadata['current_generation_meta_data'].get('fitness_score'))))
        self.metadata['evolution_gradient']['max'].append(copy.deepcopy(max(self.metadata['current_generation_meta_data'].get('fitness_score'))))
        Log().log(msg=f'Fitness: Max    -> {self.metadata["evolution_gradient"].get("max")[-1]}')
        Log().log(msg=f'Fitness: Median -> {self.metadata["evolution_gradient"].get("median")[-1]}')
        Log().log(msg=f'Fitness: Mean   -> {self.metadata["evolution_gradient"].get("mean")[-1]}')
        Log().log(msg=f'Fitness: Min    -> {self.metadata["evolution_gradient"].get("max")[-1]}')

    def generate_metadata_template(self,
                                   algorithm: str,
                                   max_iterations: int,
                                   pop_size: float,
                                   burn_in_iterations: int,
                                   warm_start: bool,
                                   change_rate: float,
                                   change_prob: float,
                                   parents_ratio: float,
                                   early_stopping: int,
                                   convergence: bool,
                                   convergence_measure: str,
                                   timer_in_seconds: int,
                                   target: str,
                                   features: List[str],
                                   models: List[str],
                                   train_data_file_path: str,
                                   test_data_file_path: str,
                                   val_data_file_path: str,
                                   re_populate: bool,
                                   max_trials: int,
                                   re_populate_threshold: float
                                   ):
        """
        Generate initial evolutionary algorithm metadata template

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
        :param target:
        :param features:
        :param models: list
        :param train_data_file_path:
        :param test_data_file_path:
        :param val_data_file_path:
        :param re_populate:
        :param max_trials:
        :param re_populate_threshold: float

        :return: dict
            Initially configured metadata template
        """
        self.metadata = dict(algorithm=algorithm,
                             current_iteration_algorithm=[],
                             max_iterations=max_iterations,
                             pop_size=pop_size,
                             burn_in_iterations=burn_in_iterations,
                             warm_start=warm_start,
                             change_rate=change_rate,
                             change_prob=change_prob,
                             parents_ratio=parents_ratio,
                             early_stopping=early_stopping,
                             stopping_reason=[],
                             convergence=convergence,
                             convergence_measure=convergence_measure,
                             timer_in_seconds=timer_in_seconds,
                             current_iteration=0,
                             parents_idx=[],
                             children_idx=[],
                             target=target,
                             features=features,
                             models=models,
                             train_data_file_path=train_data_file_path,
                             test_data_file_path=test_data_file_path,
                             val_data_file_path=val_data_file_path,
                             re_populate=re_populate,
                             max_trials=max_trials,
                             n_trial=0,
                             re_populate_threshold=re_populate_threshold,
                             labels=None,
                             best_local_idx=[],
                             best_global_idx=[],
                             generated_individuals=[],
                             generated_warm_start_individual=False,
                             current_iteration_meta_data=dict(iteration=0,
                                                              id=[],
                                                              fitness_metric=[],
                                                              fitness_score=[],
                                                              model_name=[],
                                                              param=[],
                                                              param_change=[]
                                                              ),
                             iteration_history=dict(population=dict(id=[],
                                                                    model=[],
                                                                    parent=[],
                                                                    fitness=[]
                                                                    ),
                                                    inheritance={},
                                                    time=[]
                                                    ),
                             evolution_history=dict(id=[],
                                                    model_name=[],
                                                    iteration=[],
                                                    #training=[],
                                                    parent=[],
                                                    change_type=[],
                                                    fitness_score=[],
                                                    ml_metric=[],
                                                    train_test_diff=[],
                                                    train_time_in_seconds=[],
                                                    original_ml_train_metric=[],
                                                    original_ml_test_metric=[]
                                                    ),
                             evolution_gradient=dict(min=[],
                                                     median=[],
                                                     mean=[],
                                                     max=[]
                                                     ),
                             start_time=[datetime.now()],
                             end_time=[]
                             )

    def main(self) -> List[dict]:
        """
        Apply evolutionary algorithm
        """
        self._set_iteration_algorithm()
        if self.metadata['current_iteration_algorithm'][-1] == 'ga':
            return self._mating_pool()
        elif self.metadata['current_iteration_algorithm'][-1] == 'si':
            return self._adjust_swarm()
        else:
            raise EvolutionaryAlgorithmException(f'Evolutionary algorithm ({self.metadata["current_iteration_algorithm"][-1]}) not supported')

    def populate(self) -> List[dict]:
        """
        Populate environment initially

        :return: List[dict]
        """
        _population: List[dict] = []
        _new_id: int = -1
        for idx in range(0, self.metadata['pop_size'], 1):
            _new_id += 1
            self.metadata['generated_individuals'].append(_new_id)
            if self.metadata['warm_start'] and not self.metadata['generated_warm_start_individual']:
                _warm_start: int = 1
                self.metadata['generated_warm_start_individual'] = True
            else:
                _warm_start: int = 0
            _population_instructions: dict = dict(idx=idx,
                                                  id=_new_id,
                                                  parent=None,
                                                  model_name=random.choice([self.metadata['models']]),
                                                  params=None,
                                                  param_rate=0.0,
                                                  warm_start=_warm_start
                                                  )
            _population.append(_population_instructions)
            Log().log(msg=f'Define initial model for individual {idx}')
        return _population
