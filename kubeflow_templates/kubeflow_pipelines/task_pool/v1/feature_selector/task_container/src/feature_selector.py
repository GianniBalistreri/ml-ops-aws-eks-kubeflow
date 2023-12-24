"""

Feature selector of structured (tabular) data used in supervised machine learning based on feature importance measurement

"""

import copy
import numpy as np
import os
import pandas as pd
import random

from custom_logger import Log
from evaluate_machine_learning import sml_fitness_score
from typing import Dict, List


class FeatureSelectorException(Exception):
    """
    Class for handling exceptions for class FeatureSelector
    """
    pass


class FeatureSelector:
    """
    Class for calculating shapley values (shapley additive explanations) for feature importance evaluation and feature selection
    """
    def __init__(self,
                 target_feature: str,
                 features: List[str],
                 train_df: pd.DataFrame,
                 test_df: pd.DataFrame,
                 model: object,
                 ml_type: str,
                 init_pairs: int = 3,
                 init_games: int = 5,
                 increasing_pair_size_factor: float = 0.5,
                 games: int = 3,
                 penalty_factor: float = 0.1,
                 max_iter: int = 50,
                 max_players: int = -1,
                 redundant_threshold: float = 0.01,
                 ):
        """
        :param target_feature: str
            Name of the target feature

        :param features: List[str]
            Name of the features

        :param train_df: pd.DataFrame
            Training data set

        :param test_df: pd.DataFrame
            Testing data set

        :param model: object
            Instanced model object of a decision tree

        :param ml_type: str
            Abbreviated name of the machine learning type
                -> clf_binary: Binary classification
                -> clf_multi: Multi-Classification
                -> reg: Regression

        :param init_pairs: int
            Number of players in each starting game of the tournament

        :param init_games: int
            Number of penalty games to qualify players for the tournament

        :param increasing_pair_size_factor: float
            Factor for increasing amount of player in each game in each step

        :param games: int
            Number of games to play in each step of the tournament

        :param penalty_factor: float
            Amount of players to exclude from the tournament because of their poor contribution capabilities

        :param max_iter: int
            Maximum number of steps of the tournament

        :param max_players: int
            Maximum number of features used for training machine learning model

        :param redundant_threshold: float
            Threshold for defining metric reduction to define redundant features
        """
        self.target_feature: str = target_feature
        self.features: List[str] = features
        self.n_features: int = len(self.features)
        self.train_df: pd.DataFrame = train_df
        self.test_df: pd.DataFrame = test_df
        self.n_cases: int = self.train_df.shape[0]
        self.model: object = model
        self.ml_type: str = ml_type
        if self.ml_type not in ['clf_binary', 'clf_multi', 'reg']:
            raise FeatureSelectorException(f'ML type ({self.ml_type}) not supported')
        if self.ml_type == 'reg':
            self.ml_metric: str = 'rmse_norm'
        elif self.ml_type == 'clf_binary':
            self.ml_metric: str = 'roc_auc'
        elif self.ml_type == 'clf_multi':
            self.ml_metric: str = 'cohen_kappa'
        self.init_pairs: int = init_pairs
        self.init_games: int = init_games
        self.pair_size_factor: float = increasing_pair_size_factor
        self.game: int = 0
        self.games: int = games
        self.penalty_factor: float = penalty_factor
        self.max_iter: int = max_iter
        self.max_players: int = max_players if max_players > 1 else len(self.features)
        self.redundant_threshold: float = redundant_threshold
        self.pairs: List[np.array] = []
        self.tournament: bool = False
        self.shapley_additive_explanation: dict = dict(sum={}, game={}, tournament={})
        self.imp_score: Dict[str, float] = {}
        self.plot: dict = {}

    def _feature_addition(self, imp_features: List[str]) -> dict:
        """
        Apply feature addition algorithm to select most important features

        :param imp_features: List[str]
            Name of features sorted by importance score

        :return: dict

        """
        Log().log(msg='Apply feature addition algorithm for feature selection based on calculated feature importance score')
        _model_generator: object = copy.deepcopy(self.model)
        _model_generator.train(x=self.train_df[imp_features[0]].values, y=self.train_df[self.target_feature].values)
        _pred = _model_generator.predict(x=self.test_df[imp_features[0]].values)
        _model_generator.eval(obs=self.test_df[self.target_feature].values, pred=_pred)
        _model_test_score: float = _model_generator.fitness['test'].get(self.ml_metric)
        if self.ml_type == 'reg':
            _threshold: float = _model_test_score - (_model_test_score * self.redundant_threshold)
        else:
            _threshold: float = _model_test_score + (_model_test_score * self.redundant_threshold)
        _result: dict = dict(redundant=[],
                             important=[imp_features[0]],
                             gain={},
                             model_metric=[],
                             base_metric=_model_test_score,
                             threshold=_threshold
                             )
        for i in range(1, len(imp_features) - 1, 1):
            _model_generator.train(x=self.train_df[imp_features[0:i + 1]].values, y=self.train_df[self.target_feature].values)
            _pred = _model_generator.predict(x=self.test_df[imp_features[0:i + 1]].values)
            _model_generator.eval(obs=self.test_df[self.target_feature].values, pred=_pred)
            _new_model_test_score: float = _model_generator.fitness['test'].get(self.ml_metric)
            if self.ml_type == 'reg':
                if _threshold <= _new_model_test_score:
                    _result['threshold'] = _threshold
                    _result['important'].extend(imp_features[0:i + 1])
                    _result['redundant'].extend(imp_features[i + 1:len(imp_features)])
                    break
                else:
                    _threshold: float = _new_model_test_score - (_new_model_test_score * self.redundant_threshold)
                    _result['model_metric'].append(_model_test_score)
                    _result['gain'].update({imp_features[i]: _model_test_score - _new_model_test_score})
            else:
                if _threshold >= _new_model_test_score:
                    _result['threshold'] = _threshold
                    _result['important'].extend(imp_features[0:i + 1])
                    _result['redundant'].extend(imp_features[i + 1:len(imp_features)])
                    break
                else:
                    _threshold: float = _new_model_test_score + (_new_model_test_score * self.redundant_threshold)
                    _result['model_metric'].append(_model_test_score)
                    _result['gain'].update({imp_features[i]: _new_model_test_score - _model_test_score})
        return _result

    def _filter_based(self, imp_features: List[str]) -> dict:
        """
        Apply filter-based algorithm to select most important features

        :param imp_features: List[str]
            Name of features sorted by importance score

        :return: dict

        """
        Log().log(msg='Apply filter-based algorithm for feature selection based on calculated feature importance score')
        return dict(redundant=imp_features[25:len(imp_features)], important=imp_features[0:25], reduction={}, model_metric=[], base_metric=None)

    def _game(self, iteration: int):
        """
        Play tournament game

        :param iteration: int
            Number of current iteration
        """
        for pair in self.pairs:
            _game: object = copy.deepcopy(self.model)
            _game.train(x=self.train_df[pair].values, y=self.train_df[self.target_feature].values)
            if self.ml_type == 'reg':
                _pred = _game.predict(x=self.test_df[pair].values)
                _game.eval(obs=self.test_df[self.target_feature].values, pred=_pred, train_error=False)
                _game_score: float = sml_fitness_score(ml_metric=tuple([0, _game.fitness['test'].get(self.ml_metric)]),
                                                       train_test_metric=tuple([_game.fitness['train'].get(self.ml_metric), _game.fitness['test'].get(self.ml_metric)]),
                                                       train_time_in_seconds=_game.train_time,
                                                       capping_to_zero=True
                                                       )
            else:
                _pred = _game.predict(x=self.test_df[pair].values, probability=False)
                _game.eval(obs=self.test_df[self.target_feature].values, pred=_pred, train_error=False)
                _game_score: float = sml_fitness_score(ml_metric=tuple([1, _game.fitness['test'].get(self.ml_metric)]),
                                                       train_test_metric=tuple([_game.fitness['train'].get(self.ml_metric), _game.fitness['test'].get(self.ml_metric)]),
                                                       train_time_in_seconds=_game.train_time,
                                                       capping_to_zero=True
                                                       )
            for j, imp in enumerate(_game.model.feature_importances_):
                _shapley_value: float = imp * _game_score
                if _shapley_value != _shapley_value:
                    _shapley_value = 0.0
                if pair[j] in self.shapley_additive_explanation['sum'].keys():
                    self.shapley_additive_explanation['sum'][pair[j]] += _shapley_value
                else:
                    self.shapley_additive_explanation['sum'].update({pair[j]: _shapley_value})
                if iteration >= self.init_games:
                    if pair[j] in self.shapley_additive_explanation['game'].keys():
                        self.shapley_additive_explanation['game'][pair[j]].append(_shapley_value)
                    else:
                        self.shapley_additive_explanation['game'].update({pair[j]: [_shapley_value]})

    def _permutation(self, n: int):
        """
        Permute combination of players

        :param n: int
            Number of players in each game
        """
        _shuffle: np.array = np.array(tuple(random.sample(population=self.features, k=self.n_features)))
        try:
            _pairs: np.array = np.array_split(ary=_shuffle, indices_or_sections=int(self.n_features / n))
        except ValueError:
            _pairs: np.array = self.pairs
        if self.tournament:
            for pair in _pairs:
                for feature in pair:
                    if feature in self.shapley_additive_explanation['tournament'].keys():
                        self.shapley_additive_explanation['tournament'][feature].append(len(pair))
                    else:
                        self.shapley_additive_explanation['tournament'].update({feature: [len(pair)]})
        self.pairs = _pairs

    def _play_tournament(self):
        """
        Play unreal tournament to extract the fittest or most important players based on the concept of shapley values
        """
        Log().log(msg=f'Start penalty with {self.n_features} players...')
        _game_scores: List[float] = []
        _permutation_space: int = self.init_pairs
        _pair_size_factor: float = self.max_iter * self.pair_size_factor
        for i in range(0, self.max_iter + self.init_games, 1):
            if i == self.init_games:
                Log().log(msg=f'Start feature tournament with {self.n_features} players ...')
                self.tournament = True
            elif i > self.init_games:
                _pair_size: int = _permutation_space + int(_pair_size_factor)
                if self.n_features >= _pair_size:
                    _permutation_space = _pair_size
                    #_permutation_space = int(_permutation_space + (_permutation_space * self.pair_size_factor))
            else:
                if i == 0:
                    _permutation_space = self.init_pairs
            if _permutation_space > self.max_players:
                _permutation_space = self.max_players
            self._permutation(n=_permutation_space)
            for g in range(0, self.games, 1):
                Log().log(msg=f'Iteration {i + 1} - Game {g + 1} ~ {_permutation_space} players each game')
                self.game = g
                self._game(iteration=i)
                if i < self.init_games:
                    break
                self._permutation(n=_permutation_space)
            if i + 1 == self.init_games:
                _shapley_matrix: pd.DataFrame = pd.DataFrame(data=self.shapley_additive_explanation['sum'], index=['score']).transpose()
                _sorted_shapley_matrix = _shapley_matrix.sort_values(by='score', axis=0, ascending=False, inplace=False)
                _all_features: int = _sorted_shapley_matrix.shape[0]
                _sorted_shapley_matrix = _sorted_shapley_matrix.loc[_sorted_shapley_matrix['score'] > 0, :]
                if _sorted_shapley_matrix.shape[0] == 0:
                    raise FeatureSelectorException('No feature scored higher than 0 during penalty phase')
                _n_features: int = _sorted_shapley_matrix.shape[0]
                Log().log(msg=f'Excluded {_all_features - _n_features} features with score 0')
                _exclude_features: int = int(_n_features * self.penalty_factor)
                self.features = _sorted_shapley_matrix.index.values.tolist()[0:(_n_features - _exclude_features)]
                self.n_features = len(self.features)
                Log().log(msg=f'Excluded {_exclude_features} lowest scored features from tournament')
            if i + 1 == self.max_iter + self.init_games:
                _shapley_values: dict = {}
                for sv in self.shapley_additive_explanation['game'].keys():
                    _shapley_values.update({sv: self.shapley_additive_explanation['sum'][sv] / len(self.shapley_additive_explanation['game'][sv])})
                self.shapley_additive_explanation.update({'total': _shapley_values})
            if self.n_features <= (self.pair_size_factor * _permutation_space):
                if i + 1 == self.max_iter:
                    break

    def _recursive_feature_elimination(self, imp_features: List[str]) -> dict:
        """
        Apply recursive feature elimination (RFE) algorithm to select most important features

        :param imp_features: List[str]
            Name of features sorted by importance score

        :return: dict

        """
        Log().log(msg='Apply recursive feature elimination algorithm (RFE) for feature selection based on calculated feature importance score')
        _model_generator: object = copy.deepcopy(self.model)
        _model_generator.train(x=self.train_df[imp_features].values, y=self.train_df[self.target_feature].values)
        _pred = _model_generator.predict(x=self.test_df[imp_features].values)
        _model_generator.eval(obs=self.test_df[self.target_feature].values, pred=_pred)
        _model_test_score: float = _model_generator.fitness['test'].get(self.ml_metric)
        if self.ml_type == 'reg':
            _threshold: float = _model_test_score * (1 + self.redundant_threshold)
        else:
            _threshold: float = _model_test_score * (1 - self.redundant_threshold)
        _features: List[str] = copy.deepcopy(imp_features)
        _result: dict = dict(redundant=[],
                             important=[],
                             reduction={},
                             model_metric=[],
                             base_metric=_model_test_score,
                             threshold=_threshold
                             )
        for i in range(len(imp_features) - 1, 0, -1):
            del _features[i]
            if len(_features) == 1:
                _result['important'] = _features
                break
            _model_generator.train(x=self.train_df[_features].values, y=self.train_df[self.target_feature].values)
            _pred = _model_generator.predict(x=self.test_df[_features].values)
            _model_generator.eval(obs=self.test_df[self.target_feature].values, pred=_pred)
            _new_model_test_score: float = _model_generator.fitness['test'].get(self.ml_metric)
            if self.ml_type == 'reg':
                if _threshold <= _new_model_test_score:
                    _features.append(imp_features[i])
                    _result['important'] = _features
                    break
                else:
                    _result['redundant'].append(imp_features[i])
                    _result['model_metric'].append(_model_test_score)
                    _result['reduction'].update({imp_features[i]: _model_test_score - _new_model_test_score})
            else:
                if _threshold >= _new_model_test_score:
                    _features.append(imp_features[i])
                    _result['important'] = _features
                    break
                else:
                    _result['redundant'].append(imp_features[i])
                    _result['model_metric'].append(_model_test_score)
                    _result['reduction'].update({imp_features[i]: _model_test_score - _new_model_test_score})
        return _result

    def main(self,
             feature_selection_algorithm: str = 'feature_addition',
             imp_threshold: float = 0.01,
             aggregate_feature_imp: Dict[str, dict] = None
             ) -> dict:
        """
        Select most important features based on shapley values

        :param feature_selection_algorithm: str
            Name of the feature selection algorithm to apply
                -> feature_addition: Feature Addition
                -> filter_based: Filter-based
                -> recursive_feature_elimination: Recursive Feature Elimination (RFE)

        :param imp_threshold: float
            Threshold of importance score to exclude features during initial games of the feature tournament

        :param aggregate_feature_imp: Dict[str, dict]
            Name of the aggregation method and the feature names to aggregate
                -> core: Aggregate feature importance score by each core (original) feature
                -> level: Aggregate feature importance score by the processing level of each feature

        :return dict
            Results of feature tournament like shapley values, redundant features, important features, reduction scores and evaluated metrics
        """
        self._play_tournament()
        _imp_plot: dict = {}
        _core_features: List[str] = []
        _processed_features: List[str] = []
        _imp_threshold: float = imp_threshold if (imp_threshold >= 0) and (imp_threshold <= 1) else 0.7
        _df: pd.DataFrame = pd.DataFrame(data=self.shapley_additive_explanation.get('total'), index=['score']).transpose()
        _df = _df.sort_values(by='score', axis=0, ascending=False, inplace=False)
        _df['feature'] = _df.index.values
        _imp_features: List[str] = _df['feature'].values.tolist()
        for s, feature in enumerate(_imp_features):
            self.imp_score.update({feature: _df['score'].values.tolist()[s]})
        _rank: List[int] = []
        _sorted_scores: List[float] = _df['score'].values.tolist()
        for r, val in enumerate(_sorted_scores):
            if r == 0:
                _rank.append(r + 1)
            else:
                if val == _sorted_scores[r - 1]:
                    _rank.append(_rank[-1])
                else:
                    _rank.append(r + 1)
        _df['rank'] = _rank
        self.plot.update({'Feature Importance (Shapley Scores)': dict(df=_df,
                                                                      plot_type='bar',
                                                                      render=False,
                                                                      file_path=os.path.join(str(self.path), 'feature_importance_shapley.html'),
                                                                      kwargs=dict(layout={},
                                                                                  y=_df['score'].values,
                                                                                  x=_df.index.values.tolist(),
                                                                                  marker=dict(color=_df['score'],
                                                                                              colorscale='rdylgn',
                                                                                              autocolorscale=True
                                                                                              )
                                                                                  )
                                                                      )
                          })
        _game_df: pd.DataFrame = pd.DataFrame(data=self.shapley_additive_explanation.get('game'))
        # _game_df['game'] = _game_df.index.values
        self.plot.update({'Feature Tournament Game Stats (Shapley Scores)': dict(df=_game_df,
                                                                                 features=_game_df.columns.tolist(),
                                                                                 plot_type='violin',
                                                                                 melt=True,
                                                                                 render=False,
                                                                                 file_path=os.path.join(str(self.path), 'feature_tournament_game_stats.html'),
                                                                                 kwargs=dict(layout={})
                                                                                 )
                          })
        _tournament_df: pd.DataFrame = pd.DataFrame(data=self.shapley_additive_explanation.get('tournament'))
        # _tournament_df['game'] = _tournament_df.index.values
        self.plot.update({'Feature Tournament Stats (Game Size)': dict(df=_tournament_df,
                                                                       features=_tournament_df.columns.tolist(),
                                                                       plot_type='heat',
                                                                       render=False,
                                                                       file_path=os.path.join(str(self.path), 'feature_tournament_game_size.html'),
                                                                       kwargs=dict(layout={})
                                                                       )
                          })
        if aggregate_feature_imp is not None:
            _aggre_score: dict = {}
            for core_feature in aggregate_feature_imp.keys():
                _feature_scores: dict = {}
                _aggre_score.update({core_feature: 0.0 if self.imp_score.get(core_feature) is None else self.imp_score.get(core_feature)})
                if self.imp_score.get(core_feature) is not None:
                    _feature_scores.update({core_feature: self.imp_score.get(core_feature)})
                for proc_feature in aggregate_feature_imp[core_feature]:
                    _feature_scores.update({proc_feature: 0.0 if self.imp_score.get(proc_feature) is None else self.imp_score.get(proc_feature)})
                    if self.imp_score.get(proc_feature) is not None:
                        _aggre_score[core_feature] += self.imp_score.get(proc_feature)
                if len(aggregate_feature_imp[core_feature]) < 2:
                    continue
                _aggre_score[core_feature] = _aggre_score[core_feature] / len(aggregate_feature_imp[core_feature])
                _processed_feature_matrix: pd.DataFrame = pd.DataFrame(data=_feature_scores, index=['score']).transpose()
                _processed_feature_matrix.sort_values(by='score', axis=0, ascending=False, inplace=True)
                _processed_features.append(_processed_feature_matrix.index.values.tolist()[0])
                self.plot.update({f'Feature Importance (Preprocessing Variants {core_feature})': dict(data=_processed_feature_matrix,
                                                                                                      plot_type='bar',
                                                                                                      melt=True,
                                                                                                      render=False,
                                                                                                      file_path=os.path.join(str(self.path), 'feature_importance_processing_variants.html'),
                                                                                                      kwargs=dict(layout={},
                                                                                                                  y=_processed_feature_matrix['score'].values,
                                                                                                                  x=_processed_feature_matrix.index.values,
                                                                                                                  marker=dict(color=_processed_feature_matrix['score'],
                                                                                                                              colorscale='rdylgn',
                                                                                                                              autocolorscale=True
                                                                                                                              )
                                                                                                                  )
                                                                                                      )
                                  })
            _core_imp_matrix: pd.DataFrame = pd.DataFrame(data=_aggre_score, index=['abs_score']).transpose()
            _core_imp_matrix['rel_score'] = _core_imp_matrix['abs_score'] / sum(_core_imp_matrix['abs_score'])
            _core_imp_matrix.sort_values(by='abs_score', axis=0, ascending=False, inplace=True)
            _raw_core_features: List[str] = _core_imp_matrix.loc[_core_imp_matrix['rel_score'] >= _imp_threshold, :].index.values.tolist()
            for core in _raw_core_features:
                _core_features.extend(aggregate_feature_imp[core])
                _core_features = list(set(_core_features))
            self.plot.update({'Feature Importance (Core Feature Aggregation)': dict(data=_core_imp_matrix,
                                                                                    plot_type='bar',
                                                                                    melt=False,
                                                                                    render=False,
                                                                                    file_path=os.path.join(str(self.path), 'feature_importance_core_aggregation.html'),
                                                                                    kwargs=dict(layout={},
                                                                                                y=_core_imp_matrix['abs_score'].values,
                                                                                                x=_core_imp_matrix['abs_score'].index.values,
                                                                                                marker=dict(color=_core_imp_matrix['abs_score'],
                                                                                                            colorscale='rdylgn',
                                                                                                            autocolorscale=True
                                                                                                            )
                                                                                                )
                                                                                    )
                              })
        if feature_selection_algorithm == 'feature_addition':
            _feature_selection_result: dict = self._feature_addition(imp_features=_imp_features)
        elif feature_selection_algorithm == 'filter_based':
            _feature_selection_result: dict = self._filter_based(imp_features=_imp_features)
        elif feature_selection_algorithm == 'recursive_feature_elimination':
            _feature_selection_result: dict = self._recursive_feature_elimination(imp_features=_imp_features)
        else:
            raise FeatureSelectorException(f'Feature selection algorithm ({feature_selection_algorithm}) not supported')
        Log().log(msg=f'Number of redundant features: {len(_feature_selection_result["redundant"])}\nNumber of important features: {len(_feature_selection_result["important"])}')
        return dict(imp_features=_imp_features,
                    imp_score=self.imp_score,
                    imp_threshold=imp_threshold,
                    imp_core_features=_core_features,
                    imp_processed_features=_processed_features,
                    redundant=_feature_selection_result.get('redundant'),
                    important=_feature_selection_result.get('important'),
                    reduction=_feature_selection_result.get('reduction'),
                    gain=_feature_selection_result.get('gain'),
                    model_metric=_feature_selection_result.get('model_metric'),
                    base_metric=_feature_selection_result.get('base_metric'),
                    threshold=_feature_selection_result.get('threshold')
                    )
