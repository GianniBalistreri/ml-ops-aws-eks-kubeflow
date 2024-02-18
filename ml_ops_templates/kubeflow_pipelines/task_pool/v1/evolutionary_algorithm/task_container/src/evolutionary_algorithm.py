"""

Evolutionary algorithm

"""

import copy
import numpy as np
import os
import pandas as pd
import random

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
        self.plot: dict = {}
        if self.metadata is not None:
            self.metadata['current_iteration'] += 1
            _current_time: datetime = datetime.now()
            _current_runtime_in_seconds: int = (_current_time - datetime.strptime(self.metadata['start_time'][-1], '%Y-%m-%d %H:%M:%S')).seconds
            if len(self.metadata['time_each_iteration']) == 0:
                self.metadata['time_each_iteration'].append(_current_runtime_in_seconds)
            else:
                self.metadata['time_each_iteration'].append(_current_runtime_in_seconds - self.metadata['time_each_iteration'][-1])
            if len(self.metadata['start_time']) == len(self.metadata['end_time']):
                if _current_time > datetime.strptime(self.metadata['end_time'][-1], '%Y-%m-%d %H:%M:%S'):
                    self.metadata['start_time'].append(str(_current_time).split('.')[0])
                    self.metadata['continue_evolution'] = True
                    self.metadata['max_iterations'] += self.metadata['max_iterations']

    def _adjust_swarm(self) -> List[dict]:
        """
        Adjust swarm based on POS method

        :return: List[dict]
        """
        _adjustments: List[dict] = [{} for _ in range(0, self.metadata['pop_size'], 1)]
        self._select_best_individual()
        _new_id: int = np.array(self.metadata['generated_individuals']).argmax().item()
        for idx in range(0, self.metadata['pop_size'], 1):
            if idx != self.metadata['best_global_idx'] and idx != self.metadata['best_local_idx']:
                _new_id += 1
                self.metadata['generated_individuals'].append(_new_id)
                if np.random.uniform(low=0, high=1) > self.metadata['change_prob']:
                    _adjustment_instructions: dict = dict(idx=idx,
                                                          id=_new_id,
                                                          parent=self.metadata['best_global_idx'],
                                                          model_name=random.choice(self.metadata['models']),
                                                          params=None,
                                                          param_rate=1.0,
                                                          warm_start=0,
                                                          change_type='model'
                                                          )
                    _adjustments[idx] = _adjustment_instructions
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
                                                          params=_param,#self.metadata['current_iteration_meta_data']['param'][idx],
                                                          param_rate=_param_rate,
                                                          warm_start=0,
                                                          change_type='param'
                                                          )
                    _adjustments[idx] = _adjustment_instructions
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
        _mutations: List[dict] = [{} for _ in range(0, self.metadata['pop_size'], 1)]
        self._natural_selection()
        self._inherit()
        _new_id: int = np.array(self.metadata['generated_individuals']).argmax().item()
        for parent, child in self.parent_child_pairs:
            _new_id += 1
            self.metadata['generated_individuals'].append(_new_id)
            if np.random.uniform(low=0, high=1) > self.metadata['change_prob']:
                _mutation_instructions: dict = dict(idx=child,
                                                    id=_new_id,
                                                    parent=parent,
                                                    model_name=random.choice(self.metadata['models']),
                                                    params=None,
                                                    param_rate=1.0,
                                                    warm_start=0,
                                                    change_type='model'
                                                    )
                _mutations[child] = _mutation_instructions
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
                                                    warm_start=0,
                                                    change_type='param'
                                                    )
                _mutations[child] = _mutation_instructions
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
        _best_global_idx: int = np.array(self.metadata['current_iteration_meta_data']['fitness_score']).argmax().item()
        _other_idx: List[float] = copy.deepcopy(self.metadata['current_iteration_meta_data']['fitness_score'])
        del _other_idx[_best_global_idx]
        _best_local_idx: int = np.array(_other_idx).argmax().item()
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
            _best_score: float = max(self.metadata['current_iteration_meta_data']['fitness_score'])
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
        if self.metadata['current_iteration'] > self.metadata['burn_in_iterations']:
            if self.metadata['convergence']:
                if self._is_gradient_converged(threshold=0.05):
                    _results['evolve'] = False
                    _results['stopping_reason'] = 'gradient_converged'
                    Log().log(msg=f'Fitness metric (gradient) has converged. Therefore the evolution stops at iteration {self.metadata["current_iteration_meta_data"]["iteration"]}')
            if self.metadata['early_stopping']:
                if self._is_gradient_stagnating(min_fitness=True, median_fitness=True, mean_fitness=True, max_fitness=True):
                    _results['evolve'] = False
                    _results['stopping_reason'] = 'gradient_stagnating'
                    Log().log(msg=f'Fitness metric (gradient) per iteration has not increased a certain amount of iterations ({self.metadata["early_stopping"]}). Therefore the evolution stops early at iteration {self.metadata["current_iteration_meta_data"]["iteration"]}')
        _current_runtime_in_seconds: int = (datetime.now() - datetime.strptime(self.metadata['start_time'][-1], '%Y-%m-%d %H:%M:%S')).seconds
        if _current_runtime_in_seconds >= self.metadata['timer_in_seconds']:
            _results['evolve'] = False
            _results['stopping_reason'] = 'time_exceeded'
            Log().log(msg=f'Time exceeded:{self.metadata["timer_in_seconds"]} by {_current_runtime_in_seconds} seconds')
        if self.metadata['current_iteration'] >= self.metadata['max_iterations']:
            _results['evolve'] = False
            _results['stopping_reason'] = 'max_iteration_evolved'
            Log().log(msg=f'Maximum number of iterations reached: {self.metadata["max_iterations"]}')
        if not _results.get('evolve'):
            self.metadata['stopping_reason'].append(_results['stopping_reason'])
            self.metadata['end_time'].append(str(datetime.now()).split('.')[0])
        return _results

    def gather_metadata(self, environment_reaction: dict) -> None:
        """
        Gather observed metadata of current iteration

        :param environment_reaction: dict
        """
        self.metadata['iteration_history']['time'].append((datetime.now() - datetime.strptime(self.metadata['start_time'][-1], '%Y-%m-%d %H:%M:%S')).seconds)
        if self.metadata['iteration_history']['population'].get(f'iter_{self.metadata["current_iteration"]}') is None:
            self.metadata['iteration_history']['population'].update({f'iter_{self.metadata["current_iteration"]}': dict(id=[],
                                                                                                                        model_name=[],
                                                                                                                        parent=[],
                                                                                                                        fitness=[]
                                                                                                                        )
                                                                     })
        if self.metadata['current_iteration'] == 0:
            _fittest_individual_previous_iteration: List[int] = []
        else:
            if self.metadata['current_iteration_algorithm'][-1] == 'ga':
                _fittest_individual_previous_iteration: List[int] = self.metadata['parents_idx']
            else:
                _fittest_individual_previous_iteration: List[int] = [self.metadata['best_global_idx'][-1], self.metadata['best_local_idx'][-1]]
        for i in range(0, self.metadata['pop_size'], 1):
            if i not in _fittest_individual_previous_iteration:
                # current iteration metadata:
                if self.metadata["current_iteration"] == 0:
                    self.metadata['current_iteration_meta_data']['id'].append(environment_reaction[str(i)]['id'])
                    self.metadata['current_iteration_meta_data']['model_name'].append(environment_reaction[str(i)]['model_name'])
                    self.metadata['current_iteration_meta_data']['param'].append(environment_reaction[str(i)]['param'])
                    self.metadata['current_iteration_meta_data']['param_changed'].append(environment_reaction[str(i)]['param_changed'])
                    self.metadata['current_iteration_meta_data']['fitness_metric'].append(environment_reaction[str(i)]['fitness_metric'])
                    self.metadata['current_iteration_meta_data']['fitness_score'].append(environment_reaction[str(i)]['fitness_score'])
                else:
                    self.metadata['current_iteration_meta_data']['id'][i] = copy.deepcopy(environment_reaction[str(i)]['id'])
                    self.metadata['current_iteration_meta_data']['model_name'][i] = copy.deepcopy(environment_reaction[str(i)]['model_name'])
                    self.metadata['current_iteration_meta_data']['param'][i] = copy.deepcopy(environment_reaction[str(i)]['param'])
                    self.metadata['current_iteration_meta_data']['param_changed'][i] = copy.deepcopy(environment_reaction[str(i)]['param_changed'])
                    self.metadata['current_iteration_meta_data']['fitness_metric'][i] = copy.deepcopy(environment_reaction[str(i)]['fitness_metric'])
                    self.metadata['current_iteration_meta_data']['fitness_score'][i] = copy.deepcopy(environment_reaction[str(i)]['fitness_score'])
                Log().log(f'Fitness score {environment_reaction[str(i)]["fitness_score"]} of individual {i}')
                Log().log(f'Fitness metric {environment_reaction[str(i)]["fitness_metric"]} of individual {i}')
                # iteration history:
                self.metadata['iteration_history']['population'][f'iter_{self.metadata["current_iteration"]}']['id'].append(environment_reaction[str(i)]['id'])
                self.metadata['iteration_history']['population'][f'iter_{self.metadata["current_iteration"]}']['model_name'].append(environment_reaction[str(i)]['model_name'])
                self.metadata['iteration_history']['population'][f'iter_{self.metadata["current_iteration"]}']['parent'].append(environment_reaction[str(i)]['parent'])
                self.metadata['iteration_history']['population'][f'iter_{self.metadata["current_iteration"]}']['fitness'].append(environment_reaction[str(i)]['fitness_score'])
                # evolution history:
                self.metadata['evolution_history']['id'].append(environment_reaction[str(i)]['id'])
                self.metadata['evolution_history']['iteration'].append(self.metadata['current_iteration'])
                self.metadata['evolution_history']['model_name'].append(environment_reaction[str(i)]['model_name'])
                self.metadata['evolution_history']['parent'].append(environment_reaction[str(i)]['parent'])
                self.metadata['evolution_history']['change_type'].append(environment_reaction[str(i)]['change_type'])
                self.metadata['evolution_history']['fitness_score'].append(environment_reaction[str(i)]['fitness_score'])
                self.metadata['evolution_history']['ml_metric'].append(environment_reaction[str(i)]['fitness_metric'])
                self.metadata['evolution_history']['train_test_diff'].append(environment_reaction[str(i)]['train_test_diff'])
                self.metadata['evolution_history']['train_time_in_seconds'].append(environment_reaction[str(i)]['train_time_in_seconds'])
                self.metadata['evolution_history']['original_ml_train_metric'].append(environment_reaction[str(i)]['original_ml_train_metric'])
                self.metadata['evolution_history']['original_ml_test_metric'].append(environment_reaction[str(i)]['original_ml_test_metric'])
        # evolution gradient:
        _current_iteration_fitness_scores: List[float] = self.metadata['current_iteration_meta_data']['fitness_score']
        self.metadata['evolution_gradient']['min'].append(copy.deepcopy(min(_current_iteration_fitness_scores)))
        self.metadata['evolution_gradient']['median'].append(copy.deepcopy(np.median(_current_iteration_fitness_scores)))
        self.metadata['evolution_gradient']['mean'].append(copy.deepcopy(np.mean(_current_iteration_fitness_scores)))
        self.metadata['evolution_gradient']['max'].append(copy.deepcopy(max(_current_iteration_fitness_scores)))
        Log().log(msg=f'Fitness: Max    -> {self.metadata["evolution_gradient"].get("max")[-1]}')
        Log().log(msg=f'Fitness: Median -> {self.metadata["evolution_gradient"].get("median")[-1]}')
        Log().log(msg=f'Fitness: Mean   -> {self.metadata["evolution_gradient"].get("mean")[-1]}')
        Log().log(msg=f'Fitness: Min    -> {self.metadata["evolution_gradient"].get("min")[-1]}')

    def generate_metadata_template(self,
                                   algorithm: str,
                                   max_iterations: int,
                                   pop_size: float,
                                   burn_in_iterations: int,
                                   warm_start: bool,
                                   change_rate: float,
                                   change_prob: float,
                                   parents_ratio: float,
                                   crossover: bool,
                                   early_stopping: bool,
                                   convergence: bool,
                                   convergence_measure: str,
                                   timer_in_seconds: int,
                                   re_populate: bool,
                                   re_populate_threshold: float,
                                   max_trials: int,
                                   target: str,
                                   features: List[str],
                                   models: List[str],
                                   train_data_file_path: str,
                                   test_data_file_path: str,
                                   val_data_file_path: str
                                   ):
        """
        Generate initial evolutionary algorithm metadata template

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

        :param timer_in_seconds: int
            Timer in seconds for stopping evolution

        :param re_populate: bool
            Whether to re-populate because of poor performance of the entire population or not

        :param re_populate_threshold: float
            Threshold to decide to re-populate

        :param max_trials: int
            Maximum number of trials for re-population

        :param target: str
            Name of the target feature

        :param features: List[str]
            Name of the features

        :param models: List[str]
            Abbreviated name of the models

        :param train_data_file_path: str
            Complete file path of the training data set

        :param test_data_file_path: str
            Complete file path of the test data set

        :param val_data_file_path: str
            Complete file path of the validation data set

        :return: dict
            Initially configured metadata template
        """
        self.metadata = dict(algorithm=algorithm,
                             current_iteration_algorithm=[algorithm],
                             continue_evolution=False,
                             max_iterations=max_iterations,
                             pop_size=pop_size,
                             burn_in_iterations=burn_in_iterations,
                             warm_start=warm_start,
                             change_rate=change_rate,
                             change_prob=change_prob,
                             parents_ratio=parents_ratio,
                             crossover=crossover,
                             early_stopping=early_stopping,
                             stopping_reason=[],
                             convergence=convergence,
                             convergence_measure=convergence_measure,
                             timer_in_seconds=timer_in_seconds,
                             time_each_iteration=[],
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
                             current_iteration_meta_data=dict(id=[],
                                                              fitness_metric=[],
                                                              fitness_score=[],
                                                              model_name=[],
                                                              param=[],
                                                              param_changed=[]
                                                              ),
                             iteration_history=dict(population={},
                                                    inheritance={},
                                                    time=[]
                                                    ),
                             evolution_history=dict(id=[],
                                                    model_name=[],
                                                    iteration=[],
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
                             start_time=[str(datetime.now()).split('.')[0]],
                             end_time=[]
                             )

    def generate_visualization_config(self,
                                      path: str,
                                      results_table: bool = True,
                                      model_distribution: bool = False,
                                      model_evolution: bool = True,
                                      param_distribution: bool = False,
                                      train_time_distribution: bool = True,
                                      breeding_map: bool = False,
                                      breeding_graph: bool = False,
                                      fitness_distribution: bool = True,
                                      fitness_evolution: bool = True,
                                      fitness_dimensions: bool = True,
                                      per_iteration: bool = True,
                                      ) -> None:
        """
        Generate visualization configuration

        :param path: str
            Path of the visualization files

        :param results_table: bool
            Evolution results table
                -> Table Chart

        :param model_evolution: bool
            Evolution of individuals
                -> Scatter Chart

        :param model_distribution: bool
            Distribution of used model types
                -> Bar Chart / Pie Chart

        :param param_distribution: bool
            Distribution of used model parameter combination
                -> Tree Map / Sunburst

        :param train_time_distribution: bool
            Distribution of training time
                -> Violin

        :param breeding_map: bool
            Breeding evolution as
                -> Heat Map

        :param breeding_graph: bool
            Breeding evolution as
                -> Network Graph

        :param fitness_distribution: bool
            Distribution of fitness metric
                -> Ridge Line Chart

        :param fitness_evolution: bool
            Evolution of fitness metric
                -> Line Chart

        :param fitness_dimensions: bool
            Calculated loss value for each dimension in fitness metric
                -> Radar Chart
                -> Tree Map

        :param per_iteration: bool
            Visualize results of each generation in detail or visualize just evolutionary results
        """
        _charts: dict = {}
        _evolution_history_data: pd.DataFrame = pd.DataFrame(data=self.metadata['evolution_history'])
        _m: List[str] = ['fitness_score', 'ml_metric', 'train_test_diff']
        _evolution_history_data[_m] = _evolution_history_data[_m].round(decimals=2)
        _evolution_gradient_data: pd.DataFrame = pd.DataFrame(data=self.metadata['evolution_gradient'])
        _evolution_gradient_data['iteration'] = [i for i in range(0, len(self.metadata['evolution_gradient'].get('max')), 1)]
        if results_table:
            self.plot.update({'Results of Genetic Algorithm:': dict(df=_evolution_history_data.to_dict(),
                                                                    features=_evolution_history_data.columns.to_list(),
                                                                    plot_type='table',
                                                                    file_path=os.path.join(path, 'ea_metadata_table.html'),
                                                                    )
                              })
        if model_evolution:
            self.plot.update({'Evolution of used ML Models:': dict(df=_evolution_history_data.to_dict(),
                                                                   features=['fitness_score', 'iteration'],
                                                                   color_feature='model',
                                                                   plot_type='scatter',
                                                                   melt=True,
                                                                   file_path=os.path.join(path, 'ea_model_evolution.html'),
                                                                   )
                              })
        if model_distribution:
            if self.metadata.get('models') is None or len(self.metadata.get('models')) > 1:
                self.plot.update({'Distribution of used ML Models:': dict(df=_evolution_history_data.to_dict(),
                                                                          features=['model'],
                                                                          group_by=['iteration'] if per_iteration else None,
                                                                          plot_type='pie',
                                                                          file_path=os.path.join(path, 'ea_model_distribution.html'),
                                                                          )
                                  })
        if param_distribution:
            self.plot.update({'Distribution of ML Model parameters:': dict(data=_evolution_history_data.to_dict(),
                                                                           features=['model_param'],
                                                                           group_by=['iteration'] if per_iteration else None,
                                                                           plot_type='tree',
                                                                           file_path=os.path.join(path, 'ea_parameter_treemap.html')
                                                                         )
                            })
        if train_time_distribution:
            self.plot.update({'Distribution of elapsed Training Time:': dict(df=_evolution_history_data.to_dict(),
                                                                             features=['train_time_in_seconds'],
                                                                             group_by=['model'],
                                                                             melt=True,
                                                                             plot_type='violin',
                                                                             use_auto_extensions=False,
                                                                             file_path=os.path.join(path, 'ea_training_time_distribution.html')
                                                                             )
                              })
        if breeding_map:
            _breeding_map: pd.DataFrame = pd.DataFrame(data=dict(gen_0=self.metadata['generation_history']['population']['gen_0'].get('fitness')), index=[0])
            for i in self.metadata['iteration_history']['population'].keys():
                if i != 'iter_0':
                    _breeding_map[i] = self.metadata['iteration_history']['population'][i].get('fitness')
            self.plot.update({'Breeding Heat Map:': dict(df=_breeding_map.to_dict(),
                                                         features=_breeding_map.columns.to_list(),
                                                         plot_type='heat',
                                                         file_path=os.path.join(path, 'ea_breeding_heatmap.html')
                                                         )
                              })
        if breeding_graph:
            self.plot.update({'Breeding Network Graph:': dict(df=_evolution_history_data.to_dict(),
                                                              features=['iteration', 'fitness_score'],
                                                              graph_features=dict(node='id', edge='parent'),
                                                              color_feature='model',
                                                              plot_type='network',
                                                              file_path=os.path.join(path, 'ea_breeding_graph.html')
                                                              )
                              })
        if fitness_distribution:
            self.plot.update({'Distribution of Fitness Metric:': dict(df=_evolution_history_data.to_dict(),
                                                                      features=['fitness_score'],
                                                                      time_features=['iteration'],
                                                                      plot_type='ridgeline',
                                                                      file_path=os.path.join(path, 'ea_fitness_score_distribution_per_generation.html')
                                                                      )
                              })
        if fitness_dimensions:
            self.plot.update({'Evolution Meta Data:': dict(df=_evolution_history_data.to_dict(),
                                                           features=['train_time_in_seconds',
                                                                     'ml_metric',
                                                                     'train_test_diff',
                                                                     'fitness_score',
                                                                     'parent',
                                                                     'id',
                                                                     'generation',
                                                                     'model'
                                                                     ],
                                                           color_feature='model',
                                                           plot_type='parcoords',
                                                           file_path=os.path.join(path, 'ea_metadata_evolution_coords.html')
                                                           )
                              })
        if fitness_evolution:
            self.plot.update({'Fitness Evolution:': dict(df=_evolution_gradient_data.to_dict(),
                                                         features=['min', 'median', 'mean', 'max'],
                                                         time_features=['iteration'],
                                                         melt=True,
                                                         plot_type='line',
                                                         file_path=os.path.join(path, 'ea_evolution_fitness_score.html')
                                                         )
                              })

    def main(self) -> List[dict]:
        """
        Apply evolutionary algorithm
        """
        Log().log(msg=f'Remaining time until algorithm ultimately stops: {self.metadata["timer_in_seconds"] - sum(self.metadata["time_each_iteration"])} seconds')
        Log().log(msg=f'Current iteration: {self.metadata.get("current_iteration")} (Max iterations: {self.metadata.get("max_iterations")})')
        self._set_iteration_algorithm()
        _algorithm: str = self.metadata['current_iteration_algorithm'][-1]
        Log().log(msg=f'Apply evolutionary algorithm: {_algorithm}')
        if _algorithm == 'ga':
            return self._mating_pool()
        elif _algorithm == 'si':
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
                _param_rate: float = 0.0
                self.metadata['generated_warm_start_individual'] = True
            else:
                _warm_start: int = 0
                _param_rate: float = 1.0
            _population_instructions: dict = dict(idx=idx,
                                                  id=_new_id,
                                                  parent=-1,
                                                  model_name=random.choice(self.metadata['models']),
                                                  params=None,
                                                  param_rate=1.0,
                                                  warm_start=_warm_start,
                                                  change_type='model'
                                                  )
            _population.append(_population_instructions)
            Log().log(msg=f'Define initial model for individual {idx}')
        return _population
