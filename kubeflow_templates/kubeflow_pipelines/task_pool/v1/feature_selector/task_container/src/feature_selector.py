"""

Feature selector of structured (tabular) data used in supervised machine learning based on feature importance measurement

"""

import numpy as np
import os
import pandas as pd
import random

from custom_logger import Log
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
    def __init__(self, df: pd.DataFrame, model: object):
        """
        :param df: pd.DataFrame
            Data set

        :param model: object
            Instanced model object of a decision tree
        """
        self.df: pd.DataFrame = df
        self.model: object = model

    def _game(self, iteration: int):
        """
        Play tournament game

        :param iteration: int
            Number of current iteration
        """
        for pair in self.pairs:
            if self.ml_type == 'reg':
                _game: ModelGeneratorReg = ModelGeneratorReg(model_name=self.feature_tournament_ai.get('model_name'),
                                                             reg_params=self.feature_tournament_ai.get('param')
                                                             )
                _game.generate_model()
                _game.train(x=self.train_test.get('x_train')[pair].values,
                            y=self.train_test.get('y_train').values,
                            #validation=dict(x_val=self.train_test.get('x_val')[pair].values,
                            #                y_val=self.train_test.get('y_val').values
                            #                )
                            )
                _pred = _game.predict(x=self.train_test.get('x_test')[pair].values)
                _game.eval(obs=self.train_test.get('y_test').values, pred=_pred, train_error=False)
                _game_score: float = sml_fitness_score(ml_metric=tuple([0, _game.fitness['test'].get(self.ml_metric)]),
                                                       train_test_metric=tuple([_game.fitness['train'].get(self.ml_metric), _game.fitness['test'].get(self.ml_metric)]),
                                                       train_time_in_seconds=_game.train_time,
                                                       capping_to_zero=True
                                                       )
            else:
                _game: ModelGeneratorClf = ModelGeneratorClf(model_name=self.feature_tournament_ai.get('model_name'),
                                                             clf_params=self.feature_tournament_ai.get('param')
                                                             )
                _game.generate_model()
                _game.train(x=self.train_test.get('x_train')[pair].values,
                            y=self.train_test.get('y_train').values,
                            #validation=dict(x_val=self.train_test.get('x_val')[pair].values,
                            #                y_val=self.train_test.get('y_val').values
                            #                )
                            )
                _pred = _game.predict(x=self.train_test.get('x_test')[pair].values, probability=False)
                _game.eval(obs=self.train_test.get('y_test').values, pred=_pred, train_error=False)
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
        Log(write=False, level='info').log(msg='Start penalty with {} players...'.format(self.n_features))
        _game_scores: List[float] = []
        _permutation_space: int = self.init_pairs
        _pair_size_factor: float = self.max_iter * self.pair_size_factor
        for i in range(0, self.max_iter + self.init_games, 1):
            if i == self.init_games:
                Log(write=False, level='info').log(msg='Start feature tournament with {} players ...'.format(self.n_features))
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
            for thread in self.threads.keys():
                self.threads.get(thread).get()
            if i + 1 == self.init_games:
                _shapley_matrix: pd.DataFrame = pd.DataFrame(data=self.shapley_additive_explanation['sum'], index=['score']).transpose()
                _sorted_shapley_matrix = _shapley_matrix.sort_values(by='score', axis=0, ascending=False, inplace=False)
                _all_features: int = _sorted_shapley_matrix.shape[0]
                _sorted_shapley_matrix = _sorted_shapley_matrix.loc[_sorted_shapley_matrix['score'] > 0, :]
                if _sorted_shapley_matrix.shape[0] == 0:
                    raise FeatureSelectorException('No feature scored higher than 0 during penalty phase')
                _n_features: int = _sorted_shapley_matrix.shape[0]
                Log(write=False, level='info').log(msg='Excluded {} features with score 0'.format(_all_features - _n_features))
                _exclude_features: int = int(_n_features * self.penalty_factor)
                self.features = _sorted_shapley_matrix.index.values.tolist()[0:(_n_features - _exclude_features)]
                self.n_features = len(self.features)
                Log(write=False, level='info').log(msg='Excluded {} lowest scored features from tournament'.format(_exclude_features))
            if i + 1 == self.max_iter + self.init_games:
                _shapley_values: dict = {}
                for sv in self.shapley_additive_explanation['game'].keys():
                    _shapley_values.update({sv: self.shapley_additive_explanation['sum'][sv] / len(self.shapley_additive_explanation['game'][sv])})
                self.shapley_additive_explanation.update({'total': _shapley_values})
            if self.n_features <= (self.pair_size_factor * _permutation_space):
                if i + 1 == self.max_iter:
                    break

    def main(self,
             imp_threshold: float = 0.01,
             redundant_threshold: float = 0.01,
             aggregate_feature_imp: Dict[str, dict] = None,
             visualize_feature_importance: bool = True,
             visualize_variant_scores: bool = True,
             visualize_core_feature_scores: bool = True,
             visualize_game_stats: bool = True,
             plot_type: str = 'bar'
             ) -> dict:
        """
        Select most important features based on shapley values

        :param imp_threshold: float
            Threshold of importance score to exclude features during initial games of the feature tournament

        :param redundant_threshold: float
            Threshold for defining metric reduction to define redundant features

        :param aggregate_feature_imp: Dict[str, dict]
            Name of the aggregation method and the feature names to aggregate
                -> core: Aggregate feature importance score by each core (original) feature
                -> level: Aggregate feature importance score by the processing level of each feature

        :param visualize_feature_importance: bool
            Whether to visualize feature importance scores or not

        :param visualize_variant_scores: bool
            Whether to visualize all variants of feature processing importance scores separately or not

        :param visualize_core_feature_scores: bool
            Whether to visualize summarized core feature importance scores or not

        :param visualize_game_stats: bool
            Whether to visualize game statistics or not

        :param plot_type: str
            Name of the plot type
                -> pie: Pie Chart
                -> bar: Bar Chart

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
        _game_df: pd.DataFrame = pd.DataFrame(data=self.shapley_additive_explanation.get('game'))
        # _game_df['game'] = _game_df.index.values
        _tournament_df: pd.DataFrame = pd.DataFrame(data=self.shapley_additive_explanation.get('tournament'))
        # _tournament_df['game'] = _tournament_df.index.values
        _file_paths: List[str] = []
        if visualize_game_stats:
            _file_paths.append(os.path.join(str(self.path), 'feature_tournament_game_stats.html'))
            DataVisualizer(df=_game_df,
                           title='Feature Tournament Game Stats (Shapley Scores)',
                           features=_game_df.columns.tolist(),
                           melt=True,
                           plot_type='violin',
                           file_path=_file_paths[0] if self.path is not None else None
                           ).run()
            _file_paths.append(os.path.join(str(self.path), 'feature_tournament_game_size.html'))
            DataVisualizer(df=_tournament_df,
                           title='Feature Tournament Stats (Game Size)',
                           features=_tournament_df.columns.tolist(),
                           plot_type='heat',
                           file_path=_file_paths[1] if self.path is not None else None
                           ).run()
        _file_paths.append(os.path.join(str(self.path), 'feature_importance_shapley.html'))
        if visualize_feature_importance:
            _imp_plot: dict = {'Feature Importance (Shapley Scores)': dict(df=_df,
                                                                           plot_type=plot_type,
                                                                           render=True if self.path is None else False,
                                                                           file_path=_file_paths[-1] if self.path is not None else None,
                                                                           kwargs=dict(layout={},
                                                                                       y=_df['score'].values,
                                                                                       x=_df.index.values.tolist(),
                                                                                       marker=dict(color=_df['score'],
                                                                                                   colorscale='rdylgn',
                                                                                                   autocolorscale=True
                                                                                                   )
                                                                                       )
                                                                           )
                               }
            DataVisualizer(subplots=_imp_plot,
                           height=500,
                           width=500
                           ).run()
        if self.mlflow_log:
            self._mlflow_tracking(stats={'Feature Score (Feature Tournament)': _df,
                                         'Game Size (Feature Tournament)': _tournament_df,
                                         'Game Score (Feature Tournament)': _game_df
                                         },
                                  file_paths=_file_paths)
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
                if visualize_variant_scores:
                    _variant_scores = {'Feature Importance (Preprocessing Variants {})'.format(core_feature): dict(
                        data=_processed_feature_matrix,
                        plot_type=plot_type,
                        melt=True,
                        render=True if self.path is None else False,
                        file_path='{}feature_importance_processing_variants.html'.format(
                            self.path) if self.path is not None else None,
                        kwargs=dict(layout={},
                                    y=_processed_feature_matrix['score'].values,
                                    x=_processed_feature_matrix.index.values,
                                    marker=dict(color=_processed_feature_matrix['score'],
                                                colorscale='rdylgn',
                                                autocolorscale=True
                                                )
                                    )
                    )
                    }
                    DataVisualizer(subplots=_variant_scores,
                                   height=500,
                                   width=500
                                   ).run()
            _core_imp_matrix: pd.DataFrame = pd.DataFrame(data=_aggre_score, index=['abs_score']).transpose()
            _core_imp_matrix['rel_score'] = _core_imp_matrix['abs_score'] / sum(_core_imp_matrix['abs_score'])
            _core_imp_matrix.sort_values(by='abs_score', axis=0, ascending=False, inplace=True)
            _raw_core_features: List[str] = _core_imp_matrix.loc[_core_imp_matrix['rel_score'] >= _imp_threshold, :].index.values.tolist()
            for core in _raw_core_features:
                _core_features.extend(aggregate_feature_imp[core])
                _core_features = list(set(_core_features))
            if visualize_core_feature_scores:
                _core_feature_scores_plot: dict = {'Feature Importance (Core Feature Aggregation)': dict(data=_core_imp_matrix,
                                                                                                         plot_type=plot_type,
                                                                                                         melt=False,
                                                                                                         render=True if self.path is None else False,
                                                                                                         file_path='{}feature_importance_core_aggregation.html'.format(
                                                                                                             self.path) if self.path is not None else None,
                                                                                                         kwargs=dict(layout={},
                                                                                                                     y=_core_imp_matrix[
                                                                                                                         'abs_score'].values,
                                                                                                                     x=_core_imp_matrix[
                                                                                                                         'abs_score'].index.values,
                                                                                                                     marker=dict(
                                                                                                                         color=
                                                                                                                         _core_imp_matrix[
                                                                                                                             'abs_score'],
                                                                                                                         colorscale='rdylgn',
                                                                                                                         autocolorscale=True
                                                                                                                     )
                                                                                                                     )
                                                                                                         )
                                                   }
                DataVisualizer(subplots=_core_feature_scores_plot,
                               height=500,
                               width=500
                               ).run()
        if self.ml_type == 'reg':
            _model_generator: ModelGeneratorReg = ModelGeneratorReg(model_name=self.feature_tournament_ai.get('model_name'),
                                                                    reg_params=self.feature_tournament_ai.get('param')
                                                                    )
        else:
            _model_generator: ModelGeneratorClf = ModelGeneratorClf(model_name=self.feature_tournament_ai.get('model_name'),
                                                                    clf_params=self.feature_tournament_ai.get('param')
                                                                    )
        _model_generator.generate_model()
        _train_test_split: dict = MLSampler(df=self.df,
                                            target=self.target,
                                            features=_imp_features
                                            ).train_test_sampling(validation_split=0.1)
        _model_generator.train(x=_train_test_split.get('x_train').values, y=_train_test_split.get('y_train').values)
        _pred = _model_generator.predict(x=_train_test_split.get('x_test').values)
        _model_generator.eval(obs=_train_test_split.get('y_test').values, pred=_pred)
        _model_test_score: float = _model_generator.fitness['test'].get(self.ml_metric)
        if self.ml_type == 'reg':
            _threshold: float = _model_test_score * (1 + redundant_threshold)
        else:
            _threshold: float = _model_test_score * (1 - redundant_threshold)
        _features: List[str] = copy.deepcopy(_imp_features)
        _result: dict = dict(redundant=[], important=[], reduction={}, model_metric=[])
        for i in range(len(_imp_features) - 1, 0, -1):
            del _features[i]
            if len(_features) == 1:
                _result['important'] = _features
                break
            _model_generator.train(x=_train_test_split.get('x_train')[_features].values, y=_train_test_split.get('y_train').values)
            _pred = _model_generator.predict(x=_train_test_split.get('x_test')[_features].values)
            _model_generator.eval(obs=_train_test_split.get('y_test').values, pred=_pred)
            _new_model_test_score: float = _model_generator.fitness['test'].get(self.ml_metric)
            if self.ml_type == 'reg':
                if _threshold <= _new_model_test_score:
                    _features.append(_imp_features[i])
                    _result['important'] = _features
                    break
                else:
                    _result['redundant'].append(_imp_features[i])
                    _result['model_metric'].append(_model_test_score)
                    _result['reduction'].update({_imp_features[i]: _model_test_score - _new_model_test_score})
            else:
                if _threshold >= _new_model_test_score:
                    _features.append(_imp_features[i])
                    _result['important'] = _features
                    break
                else:
                    _result['redundant'].append(_imp_features[i])
                    _result['model_metric'].append(_model_test_score)
                    _result['reduction'].update({_imp_features[i]: _model_test_score - _new_model_test_score})
        Log(write=False,
            level='info'
            ).log(msg=f'Number of redundant features: {len(_result["redundant"])}\nNumber of important features: {len(_result["important"])}')
        return dict(imp_features=_imp_features,
                    imp_score=self.imp_score,
                    imp_threshold=imp_threshold,
                    imp_core_features=_core_features,
                    imp_processed_features=_processed_features,
                    redundant=_result['redundant'],
                    important=_result['important'],
                    reduction=_result['reduction'],
                    model_metric=_result['model_metric'],
                    base_metric=_model_test_score,
                    threshold=_threshold
                    )
