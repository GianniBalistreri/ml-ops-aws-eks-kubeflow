"""

Supervised machine learning model generator (Classification & Regression)

"""

import numpy as np
import pandas as pd
import os
import warnings

from catboost import CatBoostClassifier, CatBoostRegressor
from datetime import datetime
from evaluate_machine_learning import EvalClf, EvalReg
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

warnings.simplefilter(action='ignore')

ML_ALGORITHMS: dict = dict(cat='cat_boost',
                           gbo='gradient_boosting_tree',
                           rf='random_forest',
                           xgb='extreme_gradient_boosting_tree'
                           )


class ClassificationException(Exception):
    """
    Class for handling exceptions for class Classification
    """
    pass


class Classification:
    """
    Class for handling classification algorithms
    """
    def __init__(self, clf_params: dict = None, model_name: str = 'xgb', seed: int = 1234):
        """
        :param clf_params: dict
            Pre-configured classification model parameter

        :param model_name: str
            Abbreviated name of the classification model
                -> cat: CatBoost
                -> gbo: Gradient Boosting Decision Tree
                -> rf: Random Forest
                -> xgb: Extreme Gradient Boosting Decision Tree

        :param seed: int
            Seed
        """
        self.clf_params: dict = {} if clf_params is None else clf_params
        if model_name not in ML_ALGORITHMS.keys():
            raise ClassificationException(f'Model ({model_name}) not supported')
        self.model_name: str = model_name
        self.model: object = None
        self.multi: bool = False
        self.train_time = None
        self.seed: int = 1234 if seed <= 0 else seed

    def cat_boost(self) -> CatBoostClassifier:
        """
        Config CatBoost Classifier

        :return: CatBoostClassifier
        """
        return CatBoostClassifier(n_estimators=100 if self.clf_params.get('n_estimators') is None else self.clf_params.get('n_estimators'),
                                  learning_rate=0.03 if self.clf_params.get('learning_rate') is None else self.clf_params.get('learning_rate'),
                                  depth=self.clf_params.get('depth'),
                                  l2_leaf_reg=self.clf_params.get('l2_leaf_reg'),
                                  model_size_reg=self.clf_params.get('model_size_reg'),
                                  rsm=self.clf_params.get('rsm'),
                                  loss_function=self.clf_params.get('loss_function'),
                                  border_count=self.clf_params.get('border_count'),
                                  feature_border_type=self.clf_params.get('feature_border_type'),
                                  per_float_feature_quantization=self.clf_params.get('per_float_feature_quantization'),
                                  input_borders=self.clf_params.get('input_borders'),
                                  output_borders=self.clf_params.get('output_borders'),
                                  fold_permutation_block=self.clf_params.get('fold_permutation_block'),
                                  od_pval=self.clf_params.get('od_pval'),
                                  od_wait=self.clf_params.get('od_wait'),
                                  od_type=self.clf_params.get('od_type'),
                                  nan_mode=self.clf_params.get('nan_mode'),
                                  counter_calc_method=self.clf_params.get('counter_calc_method'),
                                  leaf_estimation_iterations=self.clf_params.get('leaf_estimation_iterations'),
                                  leaf_estimation_method=self.clf_params.get('leaf_estimation_method'),
                                  thread_count=self.clf_params.get('thread_count'),
                                  random_seed=self.clf_params.get('random_seed'),
                                  use_best_model=self.clf_params.get('use_best_model'),
                                  best_model_min_trees=self.clf_params.get('best_model_min_trees'),
                                  verbose=self.clf_params.get('verbose'),
                                  silent=self.clf_params.get('silent'),
                                  logging_level=self.clf_params.get('logging_level'),
                                  metric_period=self.clf_params.get('metric_period'),
                                  ctr_leaf_count_limit=self.clf_params.get('ctr_leaf_count_limit'),
                                  store_all_simple_ctr=self.clf_params.get('store_all_simple_ctr'),
                                  max_ctr_complexity=self.clf_params.get('max_ctr_complexity'),
                                  has_time=self.clf_params.get('has_time'),
                                  allow_const_label=self.clf_params.get('allow_const_label'),
                                  target_border=self.clf_params.get('target_border'),
                                  classes_count=self.clf_params.get('classes_count'),
                                  class_weights=self.clf_params.get('class_weights'),
                                  auto_class_weights=self.clf_params.get('auto_class_weights'),
                                  class_names=self.clf_params.get('class_names'),
                                  one_hot_max_size=self.clf_params.get('one_hot_max_size'),
                                  random_strength=self.clf_params.get('random_strength'),
                                  name=self.clf_params.get('name'),
                                  ignored_features=self.clf_params.get('ignored_features'),
                                  train_dir=self.clf_params.get('train_dir'),
                                  custom_loss=self.clf_params.get('custom_loss'),
                                  custom_metric=self.clf_params.get('custom_metric'),
                                  eval_metric=self.clf_params.get('eval_metric'),
                                  bagging_temperature=self.clf_params.get('bagging_temperature'),
                                  save_snapshot=self.clf_params.get('save_snapshot'),
                                  snapshot_file=self.clf_params.get('snapshot_file'),
                                  snapshot_interval=self.clf_params.get('snapshot_interval'),
                                  fold_len_multiplier=self.clf_params.get('fold_len_multiplier'),
                                  used_ram_limit=self.clf_params.get('used_ram_limit'),
                                  gpu_ram_part=self.clf_params.get('gpu_ram_part'),
                                  pinned_memory_size=self.clf_params.get('pinned_memory_size'),
                                  allow_writing_files=self.clf_params.get('allow_writing_files'),
                                  final_ctr_computation_mode=self.clf_params.get('final_ctr_computation_mode'),
                                  approx_on_full_history=self.clf_params.get('approx_on_full_history'),
                                  boosting_type=self.clf_params.get('boosting_type'),
                                  simple_ctr=self.clf_params.get('simple_ctr'),
                                  combinations_ctr=self.clf_params.get('combinations_ctr'),
                                  per_feature_ctr=self.clf_params.get('per_feature_ctr'),
                                  ctr_description=self.clf_params.get('ctr_description'),
                                  ctr_target_border_count=self.clf_params.get('ctr_target_border_count'),
                                  task_type=self.clf_params.get('task_type'),
                                  device_config=self.clf_params.get('device_config'),
                                  devices=self.clf_params.get('devices'),
                                  bootstrap_type=self.clf_params.get('bootstrap_type'),
                                  subsample=self.clf_params.get('subsample'),
                                  mvs_reg=self.clf_params.get('mvs_reg'),
                                  sampling_unit=self.clf_params.get('sampling_unit'),
                                  sampling_frequency=self.clf_params.get('sampling_frequency'),
                                  dev_score_calc_obj_block_size=self.clf_params.get('dev_score_calc_obj_block_size'),
                                  dev_efb_max_buckets=self.clf_params.get('dev_efb_max_buckets'),
                                  sparse_features_conflict_fraction=self.clf_params.get('sparse_features_conflict_fraction'),
                                  #max_depth=self.clf_params.get('max_depth'),
                                  num_boost_round=self.clf_params.get('num_boost_round'),
                                  num_trees=self.clf_params.get('num_trees'),
                                  colsample_bylevel=self.clf_params.get('colsample_bylevel'),
                                  random_state=self.clf_params.get('random_state'),
                                  #reg_lambda=self.clf_params.get('reg_lambda'),
                                  objective=self.clf_params.get('objective'),
                                  eta=self.clf_params.get('eta'),
                                  max_bin=self.clf_params.get('max_bin'),
                                  scale_pos_weight=self.clf_params.get('scale_pos_weight'),
                                  gpu_cat_features_storage=self.clf_params.get('gpu_cat_features_storage'),
                                  data_partition=self.clf_params.get('data_partition'),
                                  metadata=self.clf_params.get('metadata'),
                                  early_stopping_rounds=self.clf_params.get('early_stopping_rounds'),
                                  cat_features=self.clf_params.get('cat_features'),
                                  grow_policy=self.clf_params.get('grow_policy'),
                                  min_data_in_leaf=self.clf_params.get('min_data_in_leaf'),
                                  min_child_samples=self.clf_params.get('min_child_samples'),
                                  max_leaves=self.clf_params.get('max_leaves'),
                                  num_leaves=self.clf_params.get('num_leaves'),
                                  score_function=self.clf_params.get('score_function'),
                                  leaf_estimation_backtracking=self.clf_params.get('leaf_estimation_backtracking'),
                                  ctr_history_unit=self.clf_params.get('ctr_history_unit'),
                                  monotone_constraints=self.clf_params.get('monotone_constraints'),
                                  feature_weights=self.clf_params.get('feature_weights'),
                                  penalties_coefficient=self.clf_params.get('penalties_coefficient'),
                                  first_feature_use_penalties=self.clf_params.get('first_feature_use_penalties'),
                                  per_object_feature_penalties=self.clf_params.get('per_object_feature_penalties'),
                                  model_shrink_rate=self.clf_params.get('model_shrink_rate'),
                                  model_shrink_mode=self.clf_params.get('model_shrink_mode'),
                                  langevin=self.clf_params.get('langevin'),
                                  diffusion_temperature=self.clf_params.get('diffusion_temperature'),
                                  posterior_sampling=self.clf_params.get('posterior_sampling'),
                                  boost_from_average=self.clf_params.get('boost_from_average'),
                                  text_features=self.clf_params.get('text_features'),
                                  tokenizers=self.clf_params.get('tokenizers'),
                                  dictionaries=self.clf_params.get('dictionaries'),
                                  feature_calcers=self.clf_params.get('feature_calcers'),
                                  text_processing=self.clf_params.get('text_processing'),
                                  embedding_features=self.clf_params.get('embedding_features')
                                  )

    def extreme_gradient_boosting_tree(self) -> XGBClassifier:
        """
        Training of the Extreme Gradient Boosting Classifier

        :return XGBClassifier:
            Model object
        """
        return XGBClassifier(max_depth=3 if self.clf_params.get('max_depth') is None else self.clf_params.get('max_depth'),
                             learning_rate=0.1 if self.clf_params.get('learning_rate') is None else self.clf_params.get('learning_rate'),
                             n_estimators=100 if self.clf_params.get('n_estimators') is None else self.clf_params.get('n_estimators'),
                             verbosity=0 if self.clf_params.get('verbosity') is None else self.clf_params.get('verbosity'),
                             objective='binary:logistic' if self.clf_params.get('objective') is None else self.clf_params.get('objective'),
                             booster='gbtree' if self.clf_params.get('booster') is None else self.clf_params.get('booster'),
                             n_jobs=os.cpu_count(),
                             gamma=0 if self.clf_params.get('gamma') is None else self.clf_params.get('gamma'),
                             min_child_weight=1 if self.clf_params.get('min_child_weight') is None else self.clf_params.get('min_child_weight'),
                             max_delta_step=0 if self.clf_params.get('max_delta_step') is None else self.clf_params.get('max_delta_step'),
                             subsample=1 if self.clf_params.get('subsample') is None else self.clf_params.get('subsample'),
                             colsample_bytree=1 if self.clf_params.get('colsample_bytree') is None else self.clf_params.get('colsample_bytree'),
                             colsample_bylevel=1 if self.clf_params.get('colsample_bylevel') is None else self.clf_params.get('colsample_bylevel'),
                             colsample_bynode=1 if self.clf_params.get('colsample_bynode') is None else self.clf_params.get('colsample_bynode'),
                             reg_alpha=0 if self.clf_params.get('reg_alpha') is None else self.clf_params.get('reg_alpha'),
                             reg_lambda=1 if self.clf_params.get('reg_lambda') is None else self.clf_params.get('reg_lambda'),
                             scale_pos_weight=1.0 if self.clf_params.get('scale_pos_weight') is None else self.clf_params.get('scale_pos_weight'),
                             base_score=0.5 if self.clf_params.get('base_score') is None else self.clf_params.get('base_score'),
                             random_state=self.seed
                             )

    def gradient_boosting_tree(self) -> GradientBoostingClassifier:
        """
        Config gradient boosting decision tree classifier

        :return GradientBoostingClassifier:
            Model object
        """
        return GradientBoostingClassifier(loss='deviance' if self.clf_params.get('loss') is None else self.clf_params.get('loss'),
                                          learning_rate=0.1 if self.clf_params.get('learning_rate') is None else self.clf_params.get('learning_rate'),
                                          n_estimators=100 if self.clf_params.get('n_estimators') is None else self.clf_params.get('n_estimators'),
                                          subsample=1.0 if self.clf_params.get('subsample') is None else self.clf_params.get('subsample'),
                                          criterion='friedman_mse' if self.clf_params.get('criterion') is None else self.clf_params.get('criterion'),
                                          min_samples_split=2 if self.clf_params.get('min_samples_split') is None else self.clf_params.get('min_samples_split'),
                                          min_samples_leaf=1 if self.clf_params.get('min_samples_leaf') is None else self.clf_params.get('min_samples_leaf'),
                                          min_weight_fraction_leaf=0 if self.clf_params.get('min_weight_fraction_leaf') is None else self.clf_params.get('min_weight_fraction_leaf'),
                                          max_depth=3 if self.clf_params.get('max_depth') is None else self.clf_params.get('max_depth'),
                                          min_impurity_decrease=0 if self.clf_params.get('min_impurity_decrease') is None else self.clf_params.get('min_impurity_decrease'),
                                          max_leaf_nodes=self.clf_params.get('max_leaf_nodes'),
                                          init=self.clf_params.get('init'),
                                          random_state=self.seed,
                                          max_features=self.clf_params.get('max_features'),
                                          verbose=0,
                                          warm_start=False if self.clf_params.get('warm_start') is None else self.clf_params.get('warm_start'),
                                          validation_fraction=0.1 if self.clf_params.get('validation_fraction') is None else self.clf_params.get('validation_fraction'),
                                          n_iter_no_change=self.clf_params.get('n_iter_no_change'),
                                          tol=0.0001 if self.clf_params.get('tol') is None else self.clf_params.get('tol'),
                                          ccp_alpha=0.0 if self.clf_params.get('ccp_alpha') is None else self.clf_params.get('ccp_alpha')
                                          )

    def random_forest(self) -> RandomForestClassifier:
        """
        Training of the Random Forest Classifier

        :return RandomForestClassifier:
            Model object
        """
        return RandomForestClassifier(n_estimators=500 if self.clf_params.get('n_estimators') is None else self.clf_params.get('n_estimators'),
                                      criterion='gini' if self.clf_params.get('criterion') is None else self.clf_params.get('criterion'),
                                      max_depth=1 if self.clf_params.get('max_depth') is None else self.clf_params.get('max_depth'),
                                      min_samples_split=2 if self.clf_params.get('min_samples_split') is None else self.clf_params.get('min_samples_split'),
                                      min_samples_leaf=1 if self.clf_params.get('min_samples_leaf') is None else self.clf_params.get('min_samples_leaf'),
                                      min_weight_fraction_leaf=0 if self.clf_params.get('min_weight_fraction_leaf') is None else self.clf_params.get('min_weight_fraction_leaf'),
                                      max_features='auto' if self.clf_params.get('max_features') is None else self.clf_params.get('max_features'),
                                      max_leaf_nodes=None if self.clf_params.get('max_leaf_nodes') is None else self.clf_params.get('max_leaf_nodes'),
                                      min_impurity_decrease=0 if self.clf_params.get('min_impurity_decrease') is None else self.clf_params.get('min_impurity_decrease'),
                                      bootstrap=True if self.clf_params.get('bootstrap') is None else self.clf_params.get('bootstrap'),
                                      oob_score=False if self.clf_params.get('oob_score') is None else self.clf_params.get('oob_score'),
                                      n_jobs=os.cpu_count(),
                                      random_state=self.seed,
                                      verbose=0 if self.clf_params.get('verbose') is None else self.clf_params.get('verbose'),
                                      warm_start=False if self.clf_params.get('warm_start') is None else self.clf_params.get('warm_start'),
                                      )

    def eval(self, obs: np.array, pred: np.array) -> float:
        """
        Evaluate supervised machine learning classification model

        :param obs: np.array
            Observed data

        :param pred: np.array
            Predictions

        :return float
            Metric value
        """
        self.multi: bool = len(pd.unique(obs)) > 2
        if self.multi:
            return EvalClf(obs=obs, pred=pred).cohen_kappa()
        else:
            return EvalClf(obs=obs, pred=pred).roc_auc()

    def generate(self) -> None:
        """
        Generate classification model
        """
        self.model = getattr(self, ML_ALGORITHMS.get(self.model_name), None)()

    def predict(self, x: np.ndarray, probability: bool = False) -> np.array:
        """
        Get prediction from trained supervised machine learning model

        :param x: np.ndarray
            Test data set

        :param probability: bool
            Calculate probability or class score

        :return np.array: Prediction
        """
        if probability:
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(x).flatten()
            else:
                raise ClassificationException(f'Model ({self.model_name}) has no function called "predict_proba"')
        else:
            if hasattr(self.model, 'predict'):
                return self.model.predict(x).flatten()
            else:
                raise ClassificationException(f'Model ({self.model_name}) has no function called "predict"')

    def train(self, x: np.ndarray, y: np.array, validation: dict = None):
        """
        Train or fit supervised machine learning model

        :param x: np.ndarray
            Train data set

        :param y: np.array
            Target data set

        :param validation: dict
        """
        _t0: datetime = datetime.now()
        if hasattr(self.model, 'fit'):
            if 'eval_set' in self.model.fit.__code__.co_varnames and validation is not None:
                if hasattr(self.model, 'fit_transform'):
                    self.model.fit_transform(x, y)
                else:
                    self.model.fit(x,
                                   y,
                                   eval_set=[(validation.get('x_val'), validation.get('y_val'))],
                                   early_stopping_rounds=np.random.randint(low=1, high=15) if self.clf_params.get('early_stopping') else None,
                                   verbose=False
                                   )
            else:
                if hasattr(self.model, 'fit_transform'):
                    self.model.fit_transform(x, y)
                else:
                    self.model.fit(x, y)
        else:
            raise ClassificationException('Training (fitting) method not supported by given model object')
        self.train_time = (datetime.now() - _t0).seconds


class RegressionException(Exception):
    """
    Class for handling exceptions for class Regression
    """
    pass


class Regression:
    """
    Class for handling regression algorithms
    """
    def __init__(self, reg_params: dict = None, model_name: str = 'xgb', seed: int = 1234):
        """
        :param reg_params: dict
            Pre-configured regression model parameter

        :param model_name: str
            Abbreviated name of the classification model
                -> cat: CatBoost
                -> gbo: Gradient Boosting Decision Tree
                -> rf: Random Forest
                -> xgb: Extreme Gradient Boosting Decision Tree

        :param seed: int
            Seed
        """
        self.reg_params: dict = {} if reg_params is None else reg_params
        if model_name not in ML_ALGORITHMS.keys():
            raise RegressionException(f'Model ({model_name}) not supported')
        self.model_name: str = model_name
        self.model: object = None
        self.train_time = None
        self.seed: int = 1234 if seed <= 0 else seed

    def cat_boost(self) -> CatBoostRegressor:
        """
        Config CatBoost Regressor

        :return: CatBoostRegressor
        """
        return CatBoostRegressor(n_estimators=100 if self.reg_params.get('n_estimators') is None else self.reg_params.get('n_estimators'),
                                 learning_rate=0.03 if self.reg_params.get('learning_rate') is None else self.reg_params.get('learning_rate'),
                                 depth=self.reg_params.get('depth'),
                                 l2_leaf_reg=self.reg_params.get('l2_leaf_reg'),
                                 model_size_reg=self.reg_params.get('model_size_reg'),
                                 rsm=self.reg_params.get('rsm'),
                                 loss_function=self.reg_params.get('loss_function'),
                                 border_count=self.reg_params.get('border_count'),
                                 feature_border_type=self.reg_params.get('feature_border_type'),
                                 per_float_feature_quantization=self.reg_params.get('per_float_feature_quantization'),
                                 input_borders=self.reg_params.get('input_borders'),
                                 output_borders=self.reg_params.get('output_borders'),
                                 fold_permutation_block=self.reg_params.get('fold_permutation_block'),
                                 od_pval=self.reg_params.get('od_pval'),
                                 od_wait=self.reg_params.get('od_wait'),
                                 od_type=self.reg_params.get('od_type'),
                                 nan_mode=self.reg_params.get('nan_mode'),
                                 counter_calc_method=self.reg_params.get('counter_calc_method'),
                                 leaf_estimation_iterations=self.reg_params.get('leaf_estimation_iterations'),
                                 leaf_estimation_method=self.reg_params.get('leaf_estimation_method'),
                                 thread_count=self.reg_params.get('thread_count'),
                                 random_seed=self.reg_params.get('random_seed'),
                                 use_best_model=self.reg_params.get('use_best_model'),
                                 best_model_min_trees=self.reg_params.get('best_model_min_trees'),
                                 verbose=self.reg_params.get('verbose'),
                                 silent=self.reg_params.get('silent'),
                                 logging_level=self.reg_params.get('logging_level'),
                                 metric_period=self.reg_params.get('metric_period'),
                                 ctr_leaf_count_limit=self.reg_params.get('ctr_leaf_count_limit'),
                                 store_all_simple_ctr=self.reg_params.get('store_all_simple_ctr'),
                                 max_ctr_complexity=self.reg_params.get('max_ctr_complexity'),
                                 has_time=self.reg_params.get('has_time'),
                                 allow_const_label=self.reg_params.get('allow_const_label'),
                                 target_border=self.reg_params.get('target_border'),
                                 one_hot_max_size=self.reg_params.get('one_hot_max_size'),
                                 random_strength=self.reg_params.get('random_strength'),
                                 name=self.reg_params.get('name'),
                                 ignored_features=self.reg_params.get('ignored_features'),
                                 train_dir=self.reg_params.get('train_dir'),
                                 custom_metric=self.reg_params.get('custom_metric'),
                                 eval_metric=self.reg_params.get('eval_metric'),
                                 bagging_temperature=self.reg_params.get('bagging_temperature'),
                                 save_snapshot=self.reg_params.get('save_snapshot'),
                                 snapshot_file=self.reg_params.get('snapshot_file'),
                                 snapshot_interval=self.reg_params.get('snapshot_interval'),
                                 fold_len_multiplier=self.reg_params.get('fold_len_multiplier'),
                                 used_ram_limit=self.reg_params.get('used_ram_limit'),
                                 gpu_ram_part=self.reg_params.get('gpu_ram_part'),
                                 pinned_memory_size=self.reg_params.get('pinned_memory_size'),
                                 allow_writing_files=self.reg_params.get('allow_writing_files'),
                                 final_ctr_computation_mode=self.reg_params.get('final_ctr_computation_mode'),
                                 approx_on_full_history=self.reg_params.get('approx_on_full_history'),
                                 boosting_type=self.reg_params.get('boosting_type'),
                                 simple_ctr=self.reg_params.get('simple_ctr'),
                                 combinations_ctr=self.reg_params.get('combinations_ctr'),
                                 per_feature_ctr=self.reg_params.get('per_feature_ctr'),
                                 ctr_description=self.reg_params.get('ctr_description'),
                                 ctr_target_border_count=self.reg_params.get('ctr_target_border_count'),
                                 task_type=self.reg_params.get('task_type'),
                                 device_config=self.reg_params.get('device_config'),
                                 devices=self.reg_params.get('devices'),
                                 bootstrap_type=self.reg_params.get('bootstrap_type'),
                                 subsample=self.reg_params.get('subsample'),
                                 mvs_reg=self.reg_params.get('mvs_reg'),
                                 sampling_unit=self.reg_params.get('sampling_unit'),
                                 sampling_frequency=self.reg_params.get('sampling_frequency'),
                                 dev_score_calc_obj_block_size=self.reg_params.get('dev_score_calc_obj_block_size'),
                                 dev_efb_max_buckets=self.reg_params.get('dev_efb_max_buckets'),
                                 sparse_features_conflict_fraction=self.reg_params.get('sparse_features_conflict_fraction'),
                                 #max_depth=self.reg_params.get('max_depth'),
                                 num_boost_round=self.reg_params.get('num_boost_round'),
                                 num_trees=self.reg_params.get('num_trees'),
                                 colsample_bylevel=self.reg_params.get('colsample_bylevel'),
                                 random_state=self.reg_params.get('random_state'),
                                 #reg_lambda=self.reg_params.get('reg_lambda'),
                                 objective=self.reg_params.get('objective'),
                                 eta=self.reg_params.get('eta'),
                                 max_bin=self.reg_params.get('max_bin'),
                                 gpu_cat_features_storage=self.reg_params.get('gpu_cat_features_storage'),
                                 data_partition=self.reg_params.get('data_partition'),
                                 metadata=self.reg_params.get('metadata'),
                                 early_stopping_rounds=self.reg_params.get('early_stopping_rounds'),
                                 cat_features=self.reg_params.get('cat_features'),
                                 grow_policy=self.reg_params.get('grow_policy'),
                                 min_data_in_leaf=self.reg_params.get('min_data_in_leaf'),
                                 min_child_samples=self.reg_params.get('min_child_samples'),
                                 max_leaves=self.reg_params.get('max_leaves'),
                                 num_leaves=self.reg_params.get('num_leaves'),
                                 score_function=self.reg_params.get('score_function'),
                                 leaf_estimation_backtracking=self.reg_params.get('leaf_estimation_backtracking'),
                                 ctr_history_unit=self.reg_params.get('ctr_history_unit'),
                                 monotone_constraints=self.reg_params.get('monotone_constraints'),
                                 feature_weights=self.reg_params.get('feature_weights'),
                                 penalties_coefficient=self.reg_params.get('penalties_coefficient'),
                                 first_feature_use_penalties=self.reg_params.get('first_feature_use_penalties'),
                                 per_object_feature_penalties=self.reg_params.get('per_object_feature_penalties'),
                                 model_shrink_rate=self.reg_params.get('model_shrink_rate'),
                                 model_shrink_mode=self.reg_params.get('model_shrink_mode'),
                                 langevin=self.reg_params.get('langevin'),
                                 diffusion_temperature=self.reg_params.get('diffusion_temperature'),
                                 posterior_sampling=self.reg_params.get('posterior_sampling'),
                                 boost_from_average=self.reg_params.get('boost_from_average')
                                 )

    def extreme_gradient_boosting_tree(self) -> XGBRegressor:
        """
        Training of the Extreme Gradient Boosting Regressor

        :return: XGBRegressor
            Model object
        """
        return XGBRegressor(max_depth=3 if self.reg_params.get('max_depth') is None else self.reg_params.get('max_depth'),
                            learning_rate=0.1 if self.reg_params.get('learning_rate') is None else self.reg_params.get('learning_rate'),
                            n_estimators=100 if self.reg_params.get('n_estimators') is None else self.reg_params.get('n_estimators'),
                            verbosity=0 if self.reg_params.get('verbosity') is None else self.reg_params.get('verbosity'),
                            objective='reg:squarederror' if self.reg_params.get('objective') is None else self.reg_params.get('objective'),
                            booster='gbtree' if self.reg_params.get('booster') is None else self.reg_params.get('booster'),
                            n_jobs=os.cpu_count(),
                            gamma=0 if self.reg_params.get('gamma') is None else self.reg_params.get('gamma'),
                            min_child_weight=1 if self.reg_params.get('min_child_weight') is None else self.reg_params.get('min_child_weight'),
                            max_delta_step=0 if self.reg_params.get('max_delta_step') is None else self.reg_params.get('max_delta_step'),
                            subsample=1 if self.reg_params.get('subsample') is None else self.reg_params.get('subsample'),
                            colsample_bytree=1 if self.reg_params.get('colsample_bytree') is None else self.reg_params.get('colsample_bytree'),
                            colsample_bylevel=1 if self.reg_params.get('colsample_bylevel') is None else self.reg_params.get('colsample_bylevel'),
                            colsample_bynode=1 if self.reg_params.get('colsample_bynode') is None else self.reg_params.get('colsample_bynode'),
                            reg_alpha=0 if self.reg_params.get('reg_alpha') is None else self.reg_params.get('reg_alpha'),
                            reg_lambda=1 if self.reg_params.get('reg_lambda') is None else self.reg_params.get('reg_lambda'),
                            scale_pos_weight=1.0 if self.reg_params.get('scale_pos_weight') is None else self.reg_params.get('scale_pos_weight'),
                            base_score=0.5 if self.reg_params.get('base_score') is None else self.reg_params.get('base_score'),
                            random_state=self.seed,
                            importance_type='gain' if self.reg_params.get('importance_type') is None else self.reg_params.get('importance_type')
                            )

    def gradient_boosting_tree(self) -> GradientBoostingRegressor:
        """
        Config gradient boosting decision tree regressor

        :return GradientBoostingRegressor
            Model object
        """
        return GradientBoostingRegressor(loss='ls' if self.reg_params.get('loss') is None else self.reg_params.get('loss'),
                                         learning_rate=0.1 if self.reg_params.get('learning_rate') is None else self.reg_params.get('learning_rate'),
                                         n_estimators=100 if self.reg_params.get('n_estimators') is None else self.reg_params.get('n_estimators'),
                                         subsample=1.0 if self.reg_params.get('subsample') is None else self.reg_params.get('subsample'),
                                         criterion='friedman_mse' if self.reg_params.get('criterion') is None else self.reg_params.get('criterion'),
                                         min_samples_split=2 if self.reg_params.get('min_samples_split') is None else self.reg_params.get('min_samples_split'),
                                         min_samples_leaf=1 if self.reg_params.get('min_samples_leaf') is None else self.reg_params.get('min_samples_leaf'),
                                         min_weight_fraction_leaf=0 if self.reg_params.get('min_weight_fraction_leaf') is None else self.reg_params.get('min_weight_fraction_leaf'),
                                         max_depth=3 if self.reg_params.get('max_depth') is None else self.reg_params.get('max_depth'),
                                         min_impurity_decrease=0 if self.reg_params.get('min_impurity_decrease') is None else self.reg_params.get('min_impurity_decrease'),
                                         init=self.reg_params.get('init'),
                                         random_state=self.seed,
                                         max_features=self.reg_params.get('max_features'),
                                         alpha=0.9 if self.reg_params.get('alpha') is None else self.reg_params.get('alpha'),
                                         verbose=0 if self.reg_params.get('verbose') is None else self.reg_params.get('verbose'),
                                         max_leaf_nodes=self.reg_params.get('max_leaf_nodes'),
                                         warm_start=False if self.reg_params.get('warm_start') is None else self.reg_params.get('warm_start'),
                                         validation_fraction=0.1 if self.reg_params.get('validation_fraction') is None else self.reg_params.get('validation_fraction'),
                                         n_iter_no_change=10 if self.reg_params.get('n_iter_no_change') is None else self.reg_params.get('n_iter_no_change'),
                                         tol=0.0001 if self.reg_params.get('tol') is None else self.reg_params.get('tol'),
                                         ccp_alpha=0.0 if self.reg_params.get('ccp_alpha') is None else self.reg_params.get('ccp_alpha')
                                         )

    def random_forest(self) -> RandomForestRegressor:
        """
        Config Random Forest Regressor

        :return: RandomForestRegressor
            Model object
        """
        return RandomForestRegressor(n_estimators=100 if self.reg_params.get('n_estimators') is None else self.reg_params.get('n_estimators'),
                                     criterion='mse' if self.reg_params.get('criterion') is None else self.reg_params.get('criterion'),
                                     max_depth=1 if self.reg_params.get('max_depth') is None else self.reg_params.get('max_depth'),
                                     min_samples_split=2 if self.reg_params.get('min_samples_split') is None else self.reg_params.get('min_samples_split'),
                                     min_samples_leaf=1 if self.reg_params.get('min_samples_leaf') is None else self.reg_params.get('min_samples_leaf'),
                                     min_weight_fraction_leaf=0 if self.reg_params.get('min_weight_fraction_leaf') is None else self.reg_params.get('min_weight_fraction_leaf'),
                                     max_features='auto' if self.reg_params.get('max_features') is None else self.reg_params.get('max_features'),
                                     max_leaf_nodes=None if self.reg_params.get('max_leaf_nodes') is None else self.reg_params.get('max_leaf_nodes'),
                                     min_impurity_decrease=0 if self.reg_params.get('min_impurity_decrease') is None else self.reg_params.get('min_impurity_decrease'),
                                     bootstrap=True if self.reg_params.get('bootstrap') is None else self.reg_params.get('bootstrap'),
                                     oob_score=False if self.reg_params.get('oob_score') is None else self.reg_params.get('oob_score'),
                                     n_jobs=os.cpu_count(),
                                     random_state=self.seed,
                                     verbose=0 if self.reg_params.get('verbose') is None else self.reg_params.get('verbose'),
                                     warm_start=False if self.reg_params.get('warm_start') is None else self.reg_params.get('warm_start'),
                                     )

    def eval(self, obs: np.array, pred: np.array) -> float:
        """
        Evaluate supervised machine learning classification model

        :param obs: np.array
            Observed train data

        :param pred: np.array
            Predictions based on train data

        :return float
            Metric value
        """
        return EvalReg(obs=obs, pred=pred).rmse_norm()

    def generate(self) -> None:
        """
        Generate regression model
        """
        self.model = getattr(self, ML_ALGORITHMS.get(self.model_name), None)()

    def predict(self, x) -> np.array:
        """
        Get prediction from trained supervised machine learning model

        :parma x: np.array
            Test data set

        :return np.array
            Predictions
        """
        if hasattr(self.model, 'predict'):
            return self.model.predict(x)
        else:
            raise RegressionException(f'Model ({self.model_name}) has no function called "predict"')

    def train(self, x: np.ndarray, y: np.array, validation: dict = None):
        """
        Train or fit supervised machine learning model

        :param x: np.ndarray
            Train data set

        :param y: np.array
            Target data set

        :param validation: dict
        """
        _t0: datetime = datetime.now()
        if hasattr(self.model, 'fit'):
            if 'eval_set' in self.model.fit.__code__.co_varnames and validation is not None:
                self.model.fit(x,
                               y,
                               eval_set=[(validation.get('x_val'), validation.get('y_val'))],
                               early_stopping_rounds=np.random.randint(low=1, high=15) if self.reg_params.get('early_stopping') else None,
                               verbose=False
                               )
            else:
                if hasattr(self.model, 'fit_transform'):
                    self.model.fit_transform(x, y)
                else:
                    self.model.fit(x, y)
        elif hasattr(self.model, 'train'):
            self.model.train(x, y)
        else:
            raise RegressionException('Training (fitting) method not supported by given model object')
        self.train_time = (datetime.now() - _t0).seconds
