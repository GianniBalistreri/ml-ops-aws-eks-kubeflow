"""

Supervised machine learning model generator (Classification & Regression)

"""

import copy
import joblib
import numpy as np
import pandas as pd
import os

from catboost import CatBoostClassifier, CatBoostRegressor
from custom_logger import Log
from datetime import datetime
from pygam import GAM
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, kneighbors_graph
from sklearn.svm import LinearSVC, LinearSVR, NuSVC, NuSVR, SVC, SVR
from typing import Dict, List
from xgboost import XGBClassifier, XGBRegressor

CLF_ALGORITHMS: Dict[str, str] = dict(ada='ada_boosting',
                                      cat='cat_boost',
                                      gbo='gradient_boosting_tree',
                                      knn='k_nearest_neighbor',
                                      lida='linear_discriminant_analysis',
                                      log='logistic_regression',
                                      qda='quadratic_discriminant_analysis',
                                      rf='random_forest',
                                      #lsvm='linear_support_vector_machine',
                                      svm='support_vector_machine',
                                      nusvm='nu_support_vector_machine',
                                      xgb='extreme_gradient_boosting_tree'
                                      )

REG_ALGORITHMS: Dict[str, str] = dict(ada='ada_boosting',
                                      cat='cat_boost',
                                      elastic='elastic_net',
                                      gam='generalized_additive_models',
                                      gbo='gradient_boosting_tree',
                                      knn='k_nearest_neighbor',
                                      lasso='lasso_regression',
                                      rf='random_forest',
                                      svm='support_vector_machine',
                                      #lsvm='linear_support_vector_machine',
                                      nusvm='nu_support_vector_machine',
                                      xgb='extreme_gradient_boosting_tree'
                                      )

CLF_STANDARD_PARAM: Dict[str, dict] = dict(ada=dict(n_estimators=50,
                                                    learning_rate=1.0
                                                    ),
                                           cat=dict(n_estimators=100,
                                                    learning_rate=0.03,
                                                    depth=None,
                                                    l2_leaf_reg=None,
                                                    feature_border_type=None,
                                                    rsm=None,
                                                    auto_class_weights=None,
                                                    grow_policy=None,
                                                    min_data_in_leaf=None
                                                    ),
                                           gbo=dict(loss='deviance',
                                                    learning_rate=0.1,
                                                    n_estimators=100,
                                                    subsample=1.0,
                                                    criterion='friedman_mse',
                                                    min_samples_split=2,
                                                    min_samples_leaf=1,
                                                    max_depth=3,
                                                    validation_fraction=0.1,
                                                    n_iter_no_change=None,
                                                    ccp_alpha=0.0
                                                    ),
                                           knn=dict(n_neighbors=5,
                                                    weights='uniform',
                                                    algorithm='auto',
                                                    p=2
                                                    ),
                                           lida=dict(solver='svd',
                                                     shrinkage=None
                                                     ),
                                           log=dict(penalty='l2',
                                                    C=1.0,
                                                    solver='saga',
                                                    max_iter=100
                                                    ),
                                           qda=dict(reg_param=0.0
                                                    ),
                                           rf=dict(n_estimators=500,
                                                   criterion='gini',
                                                   max_depth=1,
                                                   min_samples_split=2,
                                                   min_samples_leaf=1,
                                                   bootstrap=True
                                                   ),
                                           lsvm=dict(penalty='l2',
                                                     loss='squared_hinge',
                                                     C=1.0,
                                                     multi_class='ovr',
                                                     max_iter=1000
                                                     ),
                                           svm=dict(C=1.0,
                                                    kernel='rbf',
                                                    shrinking=True,
                                                    cache_size=200,
                                                    max_iter=1000,
                                                    decision_function_shape='ovr'
                                                    ),
                                           nusvm=dict(nu=0.5,
                                                      C=1.0,
                                                      kernel='rbf',
                                                      shrinking=True,
                                                      cache_size=200,
                                                      max_iter=1000,
                                                      decision_function_shape='ovr'
                                                      ),
                                           xgb=dict(learning_rate=0.1,
                                                    n_estimators=100,
                                                    max_depth=3,
                                                    gamma=0,
                                                    min_child_weight=1,
                                                    reg_alpha=0,
                                                    reg_lambda=1,
                                                    subsample=1,
                                                    colsample_bytree=1
                                                    )
                                           )

REG_STANDARD_PARAM: Dict[str, dict] = dict(ada=dict(n_estimators=50,
                                                    learning_rate=1.0,
                                                    loss='linear'
                                                    ),
                                           cat=dict(n_estimators=100,
                                                    learning_rate=0.03,
                                                    depth=None,
                                                    l2_leaf_reg=None,
                                                    feature_border_type=None,
                                                    rsm=None,
                                                    grow_policy=None,
                                                    min_data_in_leaf=None
                                                    ),
                                           elastic=dict(alpha=1.0,
                                                        l1_ratio=0.5,
                                                        fit_intercept=True,
                                                        max_iter=1000,
                                                        selection='cyclic',
                                                        ),
                                           gam=dict(max_iter=100,
                                                    tol=1e-4,
                                                    distribution='normal',
                                                    link='identity'
                                                    ),
                                           gbo=dict(loss='ls',
                                                    learning_rate=0.1,
                                                    n_estimators=100,
                                                    subsample=1.0,
                                                    criterion='friedman_mse',
                                                    min_samples_split=2,
                                                    min_samples_leaf=1,
                                                    max_depth=3,
                                                    alpha=0.9,
                                                    validation_fraction=0.1,
                                                    n_iter_no_change=10,
                                                    ccp_alpha=0.0
                                                    ),
                                           knn=dict(n_neighbors=5,
                                                    weights='uniform',
                                                    algorithm='auto',
                                                    leaf_size=30,
                                                    p=2
                                                    ),
                                           lasso=dict(alpha=1.0,
                                                      fit_intercept=True,
                                                      precompute=False,
                                                      max_iter=1000,
                                                      selection='cyclic'
                                                      ),
                                           rf=dict(n_estimators=100,
                                                   criterion='mse',
                                                   max_depth=1,
                                                   min_samples_split=2,
                                                   min_samples_leaf=1,
                                                   bootstrap=True
                                                   ),
                                           lsvm=dict(C=1.0,
                                                     loss='epsilon_insensitive',
                                                     max_iter=1000,
                                                     penalty='l1'
                                                     ),
                                           svm=dict(C=1.0,
                                                    kernel='rbf',
                                                    decision_function_shape='ovr',
                                                    shrinking=True,
                                                    cache_size=200,
                                                    max_iter=-1
                                                    ),
                                           nusvm=dict(nu=0.5,
                                                      C=1.0,
                                                      kernel='rbf',
                                                      shrinking=True,
                                                      cache_size=200,
                                                      max_iter=1000,
                                                      decision_function_shape='ovr'
                                                      ),
                                           xgb=dict(learning_rate=0.1,
                                                    n_estimators=100,
                                                    min_samples_split=None,
                                                    min_samples_leaf=None,
                                                    max_depth=3,
                                                    gamma=0,
                                                    min_child_weight=1,
                                                    reg_alpha=0,
                                                    reg_lambda=1,
                                                    subsample=1,
                                                    colsample_bytree=1
                                                    )
                                           )


class SupervisedMLException(Exception):
    """
    Class for handling exceptions for class Classification and Regression
    """
    pass


class Classification:
    """
    Class for handling classification algorithms
    """
    def __init__(self, clf_params: dict = None, seed: int = 1234, **kwargs):
        """
        :param clf_params: dict
            Pre-configured classification model parameter

        :param seed: int
            Seed

        :param kwargs: dict
            Key-word arguments
        """
        self.clf_params: dict = {} if clf_params is None else clf_params
        self.seed: int = 1234 if seed <= 0 else seed
        self.kwargs: dict = kwargs

    def ada_boosting(self) -> AdaBoostClassifier:
        """
        Config Ada Boosting algorithm

        :return AdaBoostClassifier:
            Model object
        """
        #_base_estimator = None
        #if self.clf_params.get('base_estimator') is not None:
        #    if self.clf_params.get('base_estimator') == 'base_decision_tree':
        #        _base_estimator = BaseDecisionTree
        #    elif self.clf_params.get('base_estimator') == 'decision_tree_classifier':
        #        _base_estimator = DecisionTreeClassifier
        #    elif self.clf_params.get('base_estimator') == 'extra_tree_classifier':
        #        _base_estimator = ExtraTreeClassifier
        return AdaBoostClassifier(n_estimators=self.clf_params.get('n_estimators'),
                                  learning_rate=self.clf_params.get('learning_rate'),
                                  random_state=self.seed
                                  )

    def ada_boosting_param(self) -> dict:
        """
        Generate Ada Boosting classifier parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(#base_estimator=np.random.choice(a=[None, 'base_decision_tree', 'decision_tree_classifier', 'extra_tree_classifier']),
            n_estimators=np.random.randint(low=5 if self.kwargs.get('n_estimators_low') is None else self.kwargs.get('n_estimators_low'),
                                           high=500 if self.kwargs.get('n_estimators_high') is None else self.kwargs.get('n_estimators_high')
                                           ),
            learning_rate=np.random.uniform(low=0.01 if self.kwargs.get('learning_rate_low') is None else self.kwargs.get('learning_rate_low'),
                                            high=1.0 if self.kwargs.get('learning_rate_high') is None else self.kwargs.get('learning_rate_high')
                                            )
            #algorithm=np.random.choice(a=['SAMME', 'SAMME.R'])
        )

    def cat_boost(self) -> CatBoostClassifier:
        """
        Config CatBoost Classifier

        :return: CatBoostClassifier
        """
        return CatBoostClassifier(n_estimators=self.clf_params.get('n_estimators'),
                                  learning_rate=self.clf_params.get('learning_rate'),
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

    def cat_boost_param(self) -> dict:
        """
        Generate Cat Boosting classifier parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(n_estimators=np.random.randint(low=5 if self.kwargs.get('n_estimators_low') is None else self.kwargs.get('n_estimators_low'),
                                                   high=100 if self.kwargs.get('n_estimators_high') is None else self.kwargs.get('n_estimators_high')
                                                   ),
                    learning_rate=np.random.uniform(low=0.0001 if self.kwargs.get('learning_rate_low') is None else self.kwargs.get('learning_rate_low'),
                                                    high=0.5 if self.kwargs.get('learning_rate_high') is None else self.kwargs.get('learning_rate_high')
                                                    ),
                    l2_leaf_reg=np.random.uniform(low=0.1 if self.kwargs.get('l2_leaf_reg_low') is None else self.kwargs.get('l2_leaf_reg_low'),
                                                  high=1.0 if self.kwargs.get('l2_leaf_reg_high') is None else self.kwargs.get('l2_leaf_reg_high')
                                                  ),
                    depth=np.random.randint(low=3 if self.kwargs.get('depth_low') is None else self.kwargs.get('depth_low'),
                                            high=16 if self.kwargs.get('depth_high') is None else self.kwargs.get('depth_high')
                                            ),
                    #sampling_frequency=np.random.choice(a=['PerTree', 'PerTreeLevel']),
                    #sampling_unit=np.random.choice(a=['Object', 'Group']),
                    grow_policy=np.random.choice(a=['SymmetricTree', 'Depthwise', 'Lossguide'] if self.kwargs.get('grow_policy_choice') is None else self.kwargs.get('grow_policy_choice'),),
                    min_data_in_leaf=np.random.randint(low=1 if self.kwargs.get('min_data_in_leaf_low') is None else self.kwargs.get('min_data_in_leaf_low'),
                                                       high=20 if self.kwargs.get('min_data_in_leaf_high') is None else self.kwargs.get('min_data_in_leaf_high')
                                                       ),
                    #max_leaves=np.random.randint(low=10, high=64),
                    rsm=np.random.uniform(low=0.1 if self.kwargs.get('rsm_low') is None else self.kwargs.get('rsm_low'),
                                          high=1 if self.kwargs.get('rsm_high') is None else self.kwargs.get('rsm_high')
                                          ),
                    #fold_len_multiplier=np.random.randint(low=2, high=4),
                    #approx_on_full_history=np.random.choice(a=[False, True]),
                    auto_class_weights=np.random.choice(a=[None, 'Balanced', 'SqrtBalanced'] if self.kwargs.get('auto_class_weights_choice') is None else self.kwargs.get('auto_class_weights_choice')),
                    #boosting_type=np.random.choice(a=['Ordered', 'Plain']),
                    #score_function=np.random.choice(a=['Cosine', 'L2', 'NewtonCosine', 'NewtonL2']),
                    #model_shrink_mode=np.random.choice(a=['Constant', 'Decreasing']),
                    #border_count=np.random.randint(low=1, high=65535),
                    feature_border_type=np.random.choice(a=['Median', 'Uniform', 'UniformAndQuantiles', 'MaxLogSum', 'MinEntropy', 'GreedyLogSum'] if self.kwargs.get('feature_border_type_choice') is None else self.kwargs.get('feature_border_type_choice'))
                    )

    def extreme_gradient_boosting_tree(self) -> XGBClassifier:
        """
        Training of the Extreme Gradient Boosting Classifier

        :return XGBClassifier:
            Model object
        """
        return XGBClassifier(max_depth=self.clf_params.get('max_depth'),
                             learning_rate=self.clf_params.get('learning_rate'),
                             n_estimators=self.clf_params.get('n_estimators'),
                             n_jobs=os.cpu_count(),
                             gamma=self.clf_params.get('gamma'),
                             min_child_weight=self.clf_params.get('min_child_weight'),
                             max_delta_step=self.clf_params.get('max_delta_step'),
                             subsample=self.clf_params.get('subsample'),
                             colsample_bytree=self.clf_params.get('colsample_bytree'),
                             reg_alpha=self.clf_params.get('reg_alpha'),
                             reg_lambda=self.clf_params.get('reg_lambda'),
                             random_state=self.seed
                             )

    def extreme_gradient_boosting_tree_param(self) -> dict:
        """
        Generate Extreme Gradient Boosting Decision Tree classifier parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(learning_rate=np.random.uniform(low=0.0001 if self.kwargs.get('learning_rate_low') is None else self.kwargs.get('learning_rate_low'),
                                                    high=0.5 if self.kwargs.get('learning_rate_high') is None else self.kwargs.get('learning_rate_high')
                                                    ),
                    n_estimators=np.random.randint(low=5 if self.kwargs.get('n_estimators_low') is None else self.kwargs.get('n_estimators_low'),
                                                   high=100 if self.kwargs.get('n_estimators_high') is None else self.kwargs.get('n_estimators_high')
                                                   ),
                    max_depth=np.random.randint(low=3 if self.kwargs.get('max_depth_low') is None else self.kwargs.get('max_depth_low'),
                                                high=12 if self.kwargs.get('max_depth_high') is None else self.kwargs.get('max_depth_high')
                                                ),
                    #booster=np.random.choice(a=['gbtree', 'gblinear']),
                    gamma=np.random.uniform(low=0.01 if self.kwargs.get('gamma_low') is None else self.kwargs.get('gamma_low'),
                                            high=0.99 if self.kwargs.get('gamma_high') is None else self.kwargs.get('gamma_high')
                                            ),
                    min_child_weight=np.random.randint(low=1 if self.kwargs.get('min_child_weight_low') is None else self.kwargs.get('min_child_weight_low'),
                                                       high=12 if self.kwargs.get('min_child_weight_high') is None else self.kwargs.get('min_child_weight_high')
                                                       ),
                    reg_alpha=np.random.uniform(low=0.0 if self.kwargs.get('reg_alpha_low') is None else self.kwargs.get('reg_alpha_low'),
                                                high=0.9 if self.kwargs.get('reg_alpha_high') is None else self.kwargs.get('reg_alpha_high')
                                                ),
                    reg_lambda=np.random.uniform(low=0.1 if self.kwargs.get('reg_lambda_low') is None else self.kwargs.get('reg_lambda_low'),
                                                 high=1.0 if self.kwargs.get('reg_lambda_high') is None else self.kwargs.get('reg_lambda_high')
                                                 ),
                    subsample=np.random.uniform(low=0.0 if self.kwargs.get('subsample_low') is None else self.kwargs.get('subsample_low'),
                                                high=1.0 if self.kwargs.get('subsample_high') is None else self.kwargs.get('subsample_high')
                                                ),
                    colsample_bytree=np.random.uniform(low=0.5 if self.kwargs.get('colsample_bytree_low') is None else self.kwargs.get('colsample_bytree_low'),
                                                       high=0.99 if self.kwargs.get('colsample_bytree_high') is None else self.kwargs.get('colsample_bytree_high')
                                                       ),
                    #scale_pos_weight=np.random.uniform(low=0.01, high=1.0),
                    #base_score=np.random.uniform(low=0.01, high=0.99)
                    )

    def gradient_boosting_tree(self) -> GradientBoostingClassifier:
        """
        Config gradient boosting decision tree classifier

        :return GradientBoostingClassifier:
            Model object
        """
        return GradientBoostingClassifier(loss=self.clf_params.get('loss'),
                                          learning_rate=self.clf_params.get('learning_rate'),
                                          n_estimators=self.clf_params.get('n_estimators'),
                                          subsample=self.clf_params.get('subsample'),
                                          criterion=self.clf_params.get('criterion'),
                                          min_samples_split=self.clf_params.get('min_samples_split'),
                                          min_samples_leaf=self.clf_params.get('min_samples_leaf'),
                                          max_depth=self.clf_params.get('max_depth'),
                                          random_state=self.seed,
                                          validation_fraction=self.clf_params.get('validation_fraction'),
                                          n_iter_no_change=self.clf_params.get('n_iter_no_change'),
                                          ccp_alpha=self.clf_params.get('ccp_alpha')
                                          )

    def gradient_boosting_tree_param(self) -> dict:
        """
        Generate Gradient Boosting Tree classifier parameter randomly

        :return: dict
            Parameter config
        """
        return dict(learning_rate=np.random.uniform(low=0.0001 if self.kwargs.get('learning_rate_low') is None else self.kwargs.get('learning_rate_low'),
                                                    high=0.5 if self.kwargs.get('learning_rate_high') is None else self.kwargs.get('learning_rate_high')
                                                    ),
                    loss=np.random.choice(a=['deviance', 'exponential'] if self.kwargs.get('loss_choice') is None else self.kwargs.get('loss_choice')),
                    n_estimators=np.random.randint(low=5 if self.kwargs.get('n_estimators_low') is None else self.kwargs.get('n_estimators_low'),
                                                   high=100 if self.kwargs.get('n_estimators_high') is None else self.kwargs.get('n_estimators_high')
                                                   ),
                    subsample=np.random.uniform(low=0.0 if self.kwargs.get('subsample_low') is None else self.kwargs.get('subsample_low'),
                                                high=1.0 if self.kwargs.get('subsample_high') is None else self.kwargs.get('subsample_high')
                                                ),
                    criterion=np.random.choice(a=['friedman_mse', 'mse', 'mae'] if self.kwargs.get('criterion_choice') is None else self.kwargs.get('criterion_choice')),
                    min_samples_split=np.random.randint(low=2 if self.kwargs.get('min_samples_split_low') is None else self.kwargs.get('min_samples_split_low'),
                                                        high=6 if self.kwargs.get('min_samples_split_high') is None else self.kwargs.get('min_samples_split_high')
                                                        ),
                    min_samples_leaf=np.random.randint(low=1 if self.kwargs.get('min_samples_leaf_low') is None else self.kwargs.get('min_samples_leaf_low'),
                                                       high=6 if self.kwargs.get('min_samples_leaf_high') is None else self.kwargs.get('min_samples_leaf_high')
                                                       ),
                    max_depth=np.random.randint(low=3 if self.kwargs.get('max_depth_low') is None else self.kwargs.get('max_depth_low'),
                                                high=12 if self.kwargs.get('max_depth_high') is None else self.kwargs.get('max_depth_high')
                                                ),
                    validation_fraction=np.random.uniform(low=0.05 if self.kwargs.get('validation_fraction_low') is None else self.kwargs.get('validation_fraction_low'),
                                                          high=0.4 if self.kwargs.get('validation_fraction_high') is None else self.kwargs.get('validation_fraction_high')
                                                          ),
                    n_iter_no_change=np.random.randint(low=2 if self.kwargs.get('n_iter_no_change_low') is None else self.kwargs.get('n_iter_no_change_low'),
                                                       high=10 if self.kwargs.get('n_iter_no_change_high') is None else self.kwargs.get('n_iter_no_change_high')
                                                       ),
                    ccp_alpha=np.random.uniform(low=0.0 if self.kwargs.get('ccp_alpha_low') is None else self.kwargs.get('ccp_alpha_low'),
                                                high=1.0 if self.kwargs.get('ccp_alpha_high') is None else self.kwargs.get('ccp_alpha_high')
                                                )
                    )

    def k_nearest_neighbor(self) -> KNeighborsClassifier:
        """
        Train k-nearest-neighbor (KNN) classifier

        :return KNeighborsClassifier:
            Model object
        """
        return KNeighborsClassifier(n_neighbors=self.clf_params.get('n_neighbors'),
                                    weights=self.clf_params.get('weights'),
                                    algorithm=self.clf_params.get('algorithm'),
                                    leaf_size=self.clf_params.get('leaf_size'),
                                    p=self.clf_params.get('p'),
                                    n_jobs=os.cpu_count()
                                    )

    def k_nearest_neighbor_param(self) -> dict:
        """
        Generate K-Nearest Neighbor classifier parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(n_neighbors=np.random.randint(low=2 if self.kwargs.get('n_neighbors_low') is None else self.kwargs.get('n_neighbors_low'),
                                                  high=12 if self.kwargs.get('n_neighbors_high') is None else self.kwargs.get('n_neighbors_high')
                                                  ),
                    weights=np.random.choice(a=['uniform', 'distance'] if self.kwargs.get('weights_choice') is None else self.kwargs.get('weights_choice')),
                    algorithm=np.random.choice(a=['auto', 'ball_tree', 'kd_tree', 'brute'] if self.kwargs.get('algorithm_choice') is None else self.kwargs.get('algorithm_choice')),
                    leaf_size=np.random.randint(low=15 if self.kwargs.get('leaf_size_low') is None else self.kwargs.get('leaf_size_low'),
                                                high=100 if self.kwargs.get('leaf_size_high') is None else self.kwargs.get('leaf_size_high')
                                                ),
                    p=np.random.choice(a=[1, 2, 3] if self.kwargs.get('p_choice') is None else self.kwargs.get('p_choice')),
                    #metric=np.random.choice(a=['minkowski', 'precomputed'])
                    )

    def linear_discriminant_analysis(self) -> LinearDiscriminantAnalysis:
        """
        Config linear discriminant analysis

        :return: LinearDiscriminantAnalysis:
            Model object
        """
        return LinearDiscriminantAnalysis(solver=self.clf_params.get('solver'),
                                          shrinkage=self.clf_params.get('shrinkage'),
                                          )

    def linear_discriminant_analysis_param(self) -> dict:
        """
        Generate Linear Discriminant Analysis classifier parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(shrinkage=np.random.uniform(low=0.0001 if self.kwargs.get('shrinkage_low') is None else self.kwargs.get('shrinkage_low'),
                                                high=0.9999 if self.kwargs.get('shrinkage_high') is None else self.kwargs.get('shrinkage_high')
                                                ),
                    solver=np.random.choice(a=['svd', 'eigen'] if self.kwargs.get('solver_choice') is None else self.kwargs.get('solver_choice'))
                    )

    def logistic_regression(self) -> LogisticRegression:
        """

        Training of Logistic Regression

        :return: LogisticRegression:
            Model object
        """
        return LogisticRegression(penalty=self.clf_params.get('penalty'),
                                  C=self.clf_params.get('C'),
                                  random_state=self.seed,
                                  solver=self.clf_params.get('solver'),
                                  max_iter=self.clf_params.get('max_iter'),
                                  n_jobs=os.cpu_count()
                                  )

    def logistic_regression_param(self) -> dict:
        """
        Generate Logistic Regression classifier parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(C=np.random.uniform(low=0.0001 if self.kwargs.get('C_low') is None else self.kwargs.get('C_low'),
                                        high=1.0 if self.kwargs.get('C_high') is None else self.kwargs.get('C_high')
                                        ),
                    penalty=np.random.choice(a=['l1', 'l2', 'elasticnet', 'none'] if self.kwargs.get('penalty_choice') is None else self.kwargs.get('penalty_choice')),
                    solver=np.random.choice(a=['liblinear', 'lbfgs', 'sag', 'saga', 'newton-cg'] if self.kwargs.get('solver') is None else self.kwargs.get('solver')),
                    max_iter=np.random.randint(low=5 if self.kwargs.get('max_iter_low') is None else self.kwargs.get('max_iter_low'),
                                               high=500 if self.kwargs.get('max_iter_high') is None else self.kwargs.get('max_iter_high')
                                               )
                    )

    def quadratic_discriminant_analysis(self) -> QuadraticDiscriminantAnalysis:
        """
        Generate Quadratic Discriminant Analysis classifier parameter configuration randomly

        :return QuadraticDiscriminantAnalysis:
            Model object
        """
        return QuadraticDiscriminantAnalysis(reg_param=self.clf_params.get('reg_param'))

    def quadratic_discriminant_analysis_param(self) -> dict:
        """
        Generate Quadratic Discriminant Analysis classifier parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(reg_param=np.random.uniform(low=0.0001 if self.kwargs.get('reg_param_low') is None else self.kwargs.get('reg_param_low'),
                                                high=0.9999 if self.kwargs.get('reg_param_high') is None else self.kwargs.get('reg_param_high')
                                                )
                    )

    def random_forest(self) -> RandomForestClassifier:
        """
        Training of the Random Forest Classifier

        :return RandomForestClassifier:
            Model object
        """
        return RandomForestClassifier(n_estimators=self.clf_params.get('n_estimators'),
                                      criterion=self.clf_params.get('criterion'),
                                      max_depth=self.clf_params.get('max_depth'),
                                      min_samples_split=self.clf_params.get('min_samples_split'),
                                      min_samples_leaf=self.clf_params.get('min_samples_leaf'),
                                      bootstrap=self.clf_params.get('bootstrap'),
                                      n_jobs=os.cpu_count(),
                                      random_state=self.seed,
                                      )

    def random_forest_param(self) -> dict:
        """
        Generate Random Forest classifier parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(n_estimators=np.random.randint(low=5 if self.kwargs.get('n_estimators_low') is None else self.kwargs.get('n_estimators_low'),
                                                   high=100 if self.kwargs.get('n_estimators_high') is None else self.kwargs.get('n_estimators_high')
                                                   ),
                    criterion=np.random.choice(a=['gini', 'entropy'] if self.kwargs.get('criterion_choice') is None else self.kwargs.get('criterion_choice')),
                    max_depth=np.random.randint(low=1 if self.kwargs.get('max_depth_low') is None else self.kwargs.get('max_depth_low'),
                                                high=12 if self.kwargs.get('max_depth_high') is None else self.kwargs.get('max_depth_high')
                                                ),
                    min_samples_split=np.random.randint(low=2 if self.kwargs.get('min_samples_split_low') is None else self.kwargs.get('min_samples_split_low'),
                                                        high=6 if self.kwargs.get('min_samples_split_high') is None else self.kwargs.get('min_samples_split_high')
                                                        ),
                    min_samples_leaf=np.random.randint(low=1 if self.kwargs.get('min_samples_leaf_low') is None else self.kwargs.get('min_samples_leaf_low'),
                                                       high=6 if self.kwargs.get('min_samples_leaf_high') is None else self.kwargs.get('min_samples_leaf_high')
                                                       ),
                    bootstrap=np.random.choice(a=[True, False] if self.kwargs.get('bootstrap_choice') is None else self.kwargs.get('bootstrap_choice'))
                    )

    def support_vector_machine(self) -> SVC:
        """
        Training of the Support Vector Machine Classifier

        :return SVC:
            Model object
        """
        return SVC(C=self.clf_params.get('C'),
                   kernel=self.clf_params.get('kernel'),
                   shrinking=self.clf_params.get('shrinking'),
                   cache_size=self.clf_params.get('cache_size'),
                   max_iter=self.clf_params.get('max_iter'),
                   decision_function_shape=self.clf_params.get('decision_function_shape'),
                   )

    def support_vector_machine_param(self) -> dict:
        """
        Generate Support Vector Machine classifier parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(C=np.random.uniform(low=0.0001 if self.kwargs.get('C_low') is None else self.kwargs.get('C_low'),
                                        high=1.0 if self.kwargs.get('C_high') is None else self.kwargs.get('C_high')
                                        ),
                    kernel=np.random.choice(a=['rbf', 'linear', 'poly', 'sigmoid'] if self.kwargs.get('kernel_choice') is None else self.kwargs.get('kernel_choice')), #'precomputed'
                    #gamma=np.random.choice(a=['auto', 'scale']),
                    shrinking=np.random.choice(a=[True, False] if self.kwargs.get('shrinking_choice') is None else self.kwargs.get('shrinking_choice')),
                    cache_size=np.random.randint(low=100 if self.kwargs.get('cache_size_low') is None else self.kwargs.get('cache_size_low'),
                                                 high=500 if self.kwargs.get('cache_size_high') is None else self.kwargs.get('cache_size_high')
                                                 ),
                    decision_function_shape=np.random.choice(a=['ovo', 'ovr'] if self.kwargs.get('decision_function_shape_choice') is None else self.kwargs.get('decision_function_shape_choice')),
                    max_iter=np.random.randint(low=100 if self.kwargs.get('max_iter_low') is None else self.kwargs.get('max_iter_low'),
                                               high=1000 if self.kwargs.get('max_iter_high') is None else self.kwargs.get('max_iter_high')
                                               )
                    )

    def linear_support_vector_machine(self) -> LinearSVC:
        """
        Config Linear Support Vector Machine Classifier

        :return LinearSVC:
            Model object
        """
        return LinearSVC(penalty=self.clf_params.get('penalty'),
                         loss=self.clf_params.get('loss'),
                         C=self.clf_params.get('C'),
                         multi_class=self.clf_params.get('multi_class'),
                         random_state=self.seed,
                         max_iter=self.clf_params.get('max_iter')
                         )

    def linear_support_vector_machine_param(self) -> dict:
        """
        Generate Linear Support Vector Machine classifier parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(C=np.random.uniform(low=0.0001 if self.kwargs.get('C_low') is None else self.kwargs.get('C_low'),
                                        high=1.0 if self.kwargs.get('C_high') is None else self.kwargs.get('C_high')
                                        ),
                    penalty=np.random.choice(a=['l1', 'l2'] if self.kwargs.get('penalty_choice') is None else self.kwargs.get('penalty_choice')),
                    loss=np.random.choice(a=['hinge', 'squared_hinge'] if self.kwargs.get('loss_choice') is None else self.kwargs.get('loss_choice')),
                    multi_class=np.random.choice(a=['ovr', 'crammer_singer'] if self.kwargs.get('multi_class_choice') is None else self.kwargs.get('multi_class_choice')),
                    max_iter=np.random.randint(low=100 if self.kwargs.get('max_iter_low') is None else self.kwargs.get('max_iter_low'),
                                               high=1000 if self.kwargs.get('max_iter_high') is None else self.kwargs.get('max_iter_high')
                                               )
                    )

    def nu_support_vector_machine(self) -> NuSVC:
        """
        Config Nu-Support Vector Machine Classifier

        :return NuSVC:
            Model object
        """
        return NuSVC(nu=self.clf_params.get('nu'),
                     kernel=self.clf_params.get('kernel'),
                     shrinking=self.clf_params.get('shrinking'),
                     cache_size=self.clf_params.get('cache_size'),
                     max_iter=self.clf_params.get('max_iter'),
                     decision_function_shape=self.clf_params.get('decision_function_shape'),
                     )

    def nu_support_vector_machine_param(self) -> dict:
        """
        Generate Nu-Support Vector Machine classifier parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(C=np.random.uniform(low=0.0001 if self.kwargs.get('C_low') is None else self.kwargs.get('C_low'),
                                        high=1.0 if self.kwargs.get('C_high') is None else self.kwargs.get('C_high')
                                        ),
                    nu=np.random.uniform(low=0.01 if self.kwargs.get('nu_low') is None else self.kwargs.get('nu_low'),
                                         high=0.99 if self.kwargs.get('nu_high') is None else self.kwargs.get('nu_high')
                                         ),
                    kernel=np.random.choice(a=['rbf', 'linear', 'poly', 'sigmoid'] if self.kwargs.get('kernel_choice') is None else self.kwargs.get('kernel_choice')), #'precomputed'
                    #gamma=np.random.choice(a=['auto', 'scale']),
                    shrinking=np.random.choice(a=[True, False] if self.kwargs.get('shrinking_choice') is None else self.kwargs.get('shrinking_choice')),
                    cache_size=np.random.randint(low=100 if self.kwargs.get('cache_size_low') is None else self.kwargs.get('cache_size_low'),
                                                 high=500 if self.kwargs.get('cache_size_high') is None else self.kwargs.get('cache_size_high')
                                                 ),
                    decision_function_shape=np.random.choice(a=['ovo', 'ovr'] if self.kwargs.get('decision_function_shape_choice') is None else self.kwargs.get('decision_function_shape_choice')),
                    max_iter=np.random.randint(low=100 if self.kwargs.get('max_iter_low') is None else self.kwargs.get('max_iter_low'),
                                               high=1000 if self.kwargs.get('max_iter_high') is None else self.kwargs.get('max_iter_high')
                                               )
                    )


class Regression:
    """
    Class for handling regression algorithms
    """
    def __init__(self, reg_params: dict = None, seed: int = 1234, **kwargs):
        """
        :param reg_params: dict
            Pre-configured regression model parameter

        :param seed: int
            Seed

        :param kwargs: dict
            Key-word arguments
        """
        self.reg_params: dict = {} if reg_params is None else reg_params
        self.seed: int = 1234 if seed <= 0 else seed
        self.kwargs: dict = kwargs

    def ada_boosting(self) -> AdaBoostRegressor:
        """
        Config Ada Boosting algorithm

        :return: AdaBoostRegressor
            Model object
        """
        return AdaBoostRegressor(n_estimators=self.reg_params.get('n_estimators'),
                                 learning_rate=self.reg_params.get('learning_rate'),
                                 loss=self.reg_params.get('loss'),
                                 random_state=self.seed
                                 )

    def ada_boosting_param(self) -> dict:
        """
        Generate Ada Boosting regressor parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(n_estimators=np.random.randint(low=5 if self.kwargs.get('n_estimators_low') is None else self.kwargs.get('n_estimators_low'),
                                                   high=500 if self.kwargs.get('n_estimators_high') is None else self.kwargs.get('n_estimators_high')
                                                   ),
                    learning_rate=np.random.uniform(low=0.01 if self.kwargs.get('learning_rate_low') is None else self.kwargs.get('learning_rate_low'),
                                                    high=1.0 if self.kwargs.get('learning_rate_high') is None else self.kwargs.get('learning_rate_high')
                                                    ),
                    # base_estimator=np.random.choice(a=[None, BaseDecisionTree, DecisionTreeRegressor, ExtraTreeRegressor]),
                    loss=np.random.choice(a=['linear', 'square', 'exponential'] if self.kwargs.get('loss_choice') is None else self.kwargs.get('loss_choice'))
                    )

    def cat_boost(self) -> CatBoostRegressor:
        """
        Config CatBoost Regressor

        :return: CatBoostRegressor
        """
        return CatBoostRegressor(n_estimators=self.reg_params.get('n_estimators'),
                                 learning_rate=self.reg_params.get('learning_rate'),
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

    def cat_boost_param(self) -> dict:
        """
        Generate Cat Boost regressor parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(n_estimators=np.random.randint(low=5 if self.kwargs.get('n_estimators_low') is None else self.kwargs.get('n_estimators_low'),
                                                   high=100 if self.kwargs.get('n_estimators_high') is None else self.kwargs.get('n_estimators_high')
                                                   ),
                    learning_rate=np.random.uniform(low=0.0001 if self.kwargs.get('learning_rate_low') is None else self.kwargs.get('learning_rate_low'),
                                                    high=0.5 if self.kwargs.get('learning_rate_high') is None else self.kwargs.get('learning_rate_high')
                                                    ),
                    l2_leaf_reg=np.random.uniform(low=0.1 if self.kwargs.get('l2_leaf_reg_low') is None else self.kwargs.get('l2_leaf_reg_low'),
                                                  high=1.0 if self.kwargs.get('l2_leaf_reg_high') is None else self.kwargs.get('l2_leaf_reg_high')
                                                  ),
                    depth=np.random.randint(low=3 if self.kwargs.get('depth_low') is None else self.kwargs.get('depth_low'),
                                            high=16 if self.kwargs.get('depth_high') is None else self.kwargs.get('depth_high')
                                            ),
                    #sampling_frequency=np.random.choice(a=['PerTree', 'PerTreeLevel']),
                    #sampling_unit=np.random.choice(a=['Object', 'Group']),
                    grow_policy=np.random.choice(a=['SymmetricTree', 'Depthwise', 'Lossguide'] if self.kwargs.get('grow_policy_choice') is None else self.kwargs.get('grow_policy_choice'), ),
                    min_data_in_leaf=np.random.randint(low=1 if self.kwargs.get('min_data_in_leaf_low') is None else self.kwargs.get('min_data_in_leaf_low'),
                                                       high=20 if self.kwargs.get('min_data_in_leaf_high') is None else self.kwargs.get('min_data_in_leaf_high')
                                                       ),
                    #max_leaves=np.random.randint(low=10, high=64),
                    rsm=np.random.uniform(low=0.1 if self.kwargs.get('rsm_low') is None else self.kwargs.get('rsm_low'),
                                          high=1 if self.kwargs.get('rsm_high') is None else self.kwargs.get('rsm_high')
                                          ),
                    #fold_len_multiplier=np.random.randint(low=2, high=4),
                    #approx_on_full_history=np.random.choice(a=[False, True]),
                    #boosting_type=np.random.choice(a=['Ordered', 'Plain']),
                    #score_function=np.random.choice(a=['Cosine', 'L2', 'NewtonCosine', 'NewtonL2']),
                    #model_shrink_mode=np.random.choice(a=['Constant', 'Decreasing']),
                    #border_count=np.random.randint(low=1, high=65535),
                    feature_border_type=np.random.choice(a=['Median', 'Uniform', 'UniformAndQuantiles', 'MaxLogSum', 'MinEntropy', 'GreedyLogSum'] if self.kwargs.get('feature_border_type_choice') is None else self.kwargs.get('feature_border_type_choice'))
                    )

    def elastic_net(self) -> ElasticNet:
        """
        Training of the elastic net regressor

        :return ElasticNet
            Object model
        """
        return ElasticNet(alpha=self.reg_params.get('alpha'),
                          l1_ratio=self.reg_params.get('l1_ratio'),
                          fit_intercept=self.reg_params.get('fit_intercept'),
                          max_iter=self.reg_params.get('max_iter'),
                          random_state=self.seed,
                          selection=self.reg_params.get('selection')
                          )

    def elastic_net_param(self) -> dict:
        """
        Generate Elastic Net regressor parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(alpha=np.random.uniform(low=0.0 if self.kwargs.get('alpha_low') is None else self.kwargs.get('alpha_low'),
                                            high=1.0 if self.kwargs.get('alpha_high') is None else self.kwargs.get('alpha_high')
                                            ),
                    l1_ratio=np.random.uniform(low=0.0 if self.kwargs.get('l1_ratio_low') is None else self.kwargs.get('l1_ratio_low'),
                                               high=1.0 if self.kwargs.get('l1_ratio_high') is None else self.kwargs.get('l1_ratio_high')
                                               ),
                    normalize=np.random.choice(a=[True, False] if self.kwargs.get('normalize_choice') is None else self.kwargs.get('normalize_choice')),
                    #precompute=np.random.choice(a=[True, False]),
                    max_iter=np.random.randint(low=5 if self.kwargs.get('max_iter_low') is None else self.kwargs.get('max_iter_low'),
                                               high=1000 if self.kwargs.get('max_iter_high') is None else self.kwargs.get('max_iter_high')
                                               ),
                    fit_intercept=np.random.choice(a=[True, False] if self.kwargs.get('fit_intercept_choice') is None else self.kwargs.get('fit_intercept_choice')),
                    selection=np.random.choice(a=['cyclic', 'random'] if self.kwargs.get('selection_choice') is None else self.kwargs.get('selection_choice'))
                    )

    def extreme_gradient_boosting_tree(self) -> XGBRegressor:
        """
        Training of the Extreme Gradient Boosting Regressor

        :return: XGBRegressor
            Model object
        """
        return XGBRegressor(max_depth=self.reg_params.get('max_depth'),
                            learning_rate=self.reg_params.get('learning_rate'),
                            n_estimators=self.reg_params.get('n_estimators'),
                            n_jobs=os.cpu_count(),
                            gamma=self.reg_params.get('gamma'),
                            min_child_weight=self.reg_params.get('min_child_weight'),
                            subsample=self.reg_params.get('subsample'),
                            colsample_bytree=self.reg_params.get('colsample_bytree'),
                            reg_alpha=self.reg_params.get('reg_alpha'),
                            reg_lambda=self.reg_params.get('reg_lambda'),
                            random_state=self.seed
                            )

    def extreme_gradient_boosting_tree_param(self) -> dict:
        """
        Generate Extreme Gradient Boosting Decision Tree regressor parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(learning_rate=np.random.uniform(low=0.0001 if self.kwargs.get('learning_rate_low') is None else self.kwargs.get('learning_rate_low'),
                                                    high=0.5 if self.kwargs.get('learning_rate_high') is None else self.kwargs.get('learning_rate_high')
                                                    ),
                    n_estimators=np.random.randint(low=5 if self.kwargs.get('n_estimators_low') is None else self.kwargs.get('n_estimators_low'),
                                                   high=100 if self.kwargs.get('n_estimators_high') is None else self.kwargs.get('n_estimators_high')
                                                   ),
                    min_samples_split=np.random.randint(low=2 if self.kwargs.get('min_samples_split_low') is None else self.kwargs.get('min_samples_split_low'),
                                                        high=6 if self.kwargs.get('min_samples_split_high') is None else self.kwargs.get('min_samples_split_high')
                                                        ),
                    min_samples_leaf=np.random.randint(low=1 if self.kwargs.get('min_samples_leaf_low') is None else self.kwargs.get('min_samples_leaf_low'),
                                                       high=6 if self.kwargs.get('min_samples_leaf_high') is None else self.kwargs.get('min_samples_leaf_high')
                                                       ),
                    max_depth=np.random.randint(low=3 if self.kwargs.get('max_depth_low') is None else self.kwargs.get('max_depth_low'),
                                                high=12 if self.kwargs.get('max_depth_high') is None else self.kwargs.get('max_depth_high')
                                                ),
                    #booster=np.random.choice(a=['gbtree', 'gblinear', 'gbdart']),
                    gamma=np.random.uniform(low=0.01 if self.kwargs.get('gamma_low') is None else self.kwargs.get('gamma_low'),
                                            high=0.99 if self.kwargs.get('gamma_high') is None else self.kwargs.get('gamma_high')
                                            ),
                    min_child_weight=np.random.randint(low=1 if self.kwargs.get('min_child_weight_low') is None else self.kwargs.get('min_child_weight_low'),
                                                       high=12 if self.kwargs.get('min_child_weight_high') is None else self.kwargs.get('min_child_weight_high')
                                                       ),
                    reg_alpha=np.random.uniform(low=0.0 if self.kwargs.get('reg_alpha_low') is None else self.kwargs.get('reg_alpha_low'),
                                                high=0.9 if self.kwargs.get('reg_alpha_high') is None else self.kwargs.get('reg_alpha_high')
                                                ),
                    reg_lambda=np.random.uniform(low=0.1 if self.kwargs.get('reg_lambda_low') is None else self.kwargs.get('reg_lambda_low'),
                                                 high=1.0 if self.kwargs.get('reg_lambda_high') is None else self.kwargs.get('reg_lambda_high')
                                                 ),
                    subsample=np.random.uniform(low=0.0 if self.kwargs.get('subsample_low') is None else self.kwargs.get('subsample_low'),
                                                high=1.0 if self.kwargs.get('subsample_high') is None else self.kwargs.get('subsample_high')
                                                ),
                    colsample_bytree=np.random.uniform(low=0.5 if self.kwargs.get('colsample_bytree_low') is None else self.kwargs.get('colsample_bytree_low'),
                                                       high=0.99 if self.kwargs.get('colsample_bytree_high') is None else self.kwargs.get('colsample_bytree_high')
                                                       ),
                    #scale_pos_weight=np.random.uniform(low=0.01, high=1.0),
                    #base_score=np.random.uniform(low=0.01, high=0.99),
                    #early_stopping=np.random.choice(a=[True, False] if self.kwargs.get('early_stopping_choice') is None else self.kwargs.get('early_stopping_choice'))
                    )

    def generalized_additive_models(self) -> GAM:
        """
        Config Generalized Additive Model regressor

        :return: GAM
            Model object
        """
        return GAM(max_iter=self.reg_params.get('max_iter'),
                   tol=self.reg_params.get('tol'),
                   distribution=self.reg_params.get('distribution'),
                   link=self.reg_params.get('link')
                   )

    def generalized_additive_models_param(self) -> dict:
        """
        Config Generalized Additive Model regressor

        :return: dict
            Parameter config
        """
        return dict(max_iter=np.random.randint(low=10 if self.kwargs.get('max_iter_low') is None else self.kwargs.get('max_iter_low'),
                                               high=500 if self.kwargs.get('max_iter_high') is None else self.kwargs.get('max_iter_high')
                                               ),
                    tol=np.random.uniform(low=0.00001 if self.kwargs.get('tol_low') is None else self.kwargs.get('tol_low'),
                                          high=0.001 if self.kwargs.get('tol_high') is None else self.kwargs.get('tol_high')
                                          ),
                    distribution=np.random.choice(a=['normal', 'binomial', 'poisson', 'gamma', 'invgauss'] if self.kwargs.get('distribution_choice') is None else self.kwargs.get('distribution_choice')),
                    link=np.random.choice(a=['identity', 'logit', 'log', 'inverse', 'inverse-squared'] if self.kwargs.get('link_choice') is None else self.kwargs.get('link_choice'))
                    )

    def gradient_boosting_tree(self) -> GradientBoostingRegressor:
        """
        Config gradient boosting decision tree regressor

        :return GradientBoostingRegressor
            Model object
        """
        return GradientBoostingRegressor(loss=self.reg_params.get('loss'),
                                         learning_rate=self.reg_params.get('learning_rate'),
                                         n_estimators=self.reg_params.get('n_estimators'),
                                         subsample=self.reg_params.get('subsample'),
                                         criterion=self.reg_params.get('criterion'),
                                         min_samples_split=self.reg_params.get('min_samples_split'),
                                         min_samples_leaf=self.reg_params.get('min_samples_leaf'),
                                         max_depth=self.reg_params.get('max_depth'),
                                         random_state=self.seed,
                                         alpha=self.reg_params.get('alpha'),
                                         validation_fraction=self.reg_params.get('validation_fraction'),
                                         n_iter_no_change=self.reg_params.get('n_iter_no_change'),
                                         ccp_alpha=self.reg_params.get('ccp_alpha')
                                         )

    def gradient_boosting_tree_param(self) -> dict:
        """
        Generate Gradient Boosting Tree regressor parameter randomly

        :return: dict
            Parameter config
        """
        return dict(learning_rate=np.random.uniform(low=0.0001 if self.kwargs.get('learning_rate_low') is None else self.kwargs.get('learning_rate_low'),
                                                    high=0.5 if self.kwargs.get('learning_rate_high') is None else self.kwargs.get('learning_rate_high')
                                                    ),
                    loss=np.random.choice(a=['ls', 'lad', 'huber', 'quantile']),
                    n_estimators=np.random.randint(low=5 if self.kwargs.get('n_estimators_low') is None else self.kwargs.get('n_estimators_low'),
                                                   high=100 if self.kwargs.get('n_estimators_high') is None else self.kwargs.get('n_estimators_high')
                                                   ),
                    subsample=np.random.uniform(low=0.0 if self.kwargs.get('subsample_low') is None else self.kwargs.get('subsample_low'),
                                                high=1.0 if self.kwargs.get('subsample_high') is None else self.kwargs.get('subsample_high')
                                                ),
                    criterion=np.random.choice(a=['friedman_mse', 'mse', 'mae'] if self.kwargs.get('criterion_choice') is None else self.kwargs.get('criterion_choice')),
                    min_samples_split=np.random.randint(low=2 if self.kwargs.get('min_samples_split_low') is None else self.kwargs.get('min_samples_split_low'),
                                                        high=6 if self.kwargs.get('min_samples_split_high') is None else self.kwargs.get('min_samples_split_high')
                                                        ),
                    min_samples_leaf=np.random.randint(low=1 if self.kwargs.get('min_samples_leaf_low') is None else self.kwargs.get('min_samples_leaf_low'),
                                                       high=6 if self.kwargs.get('min_samples_leaf_high') is None else self.kwargs.get('min_samples_leaf_high')
                                                       ),
                    max_depth=np.random.randint(low=3 if self.kwargs.get('max_depth_low') is None else self.kwargs.get('max_depth_low'),
                                                high=12 if self.kwargs.get('max_depth_high') is None else self.kwargs.get('max_depth_high')
                                                ),
                    validation_fraction=np.random.uniform(low=0.05 if self.kwargs.get('validation_fraction_low') is None else self.kwargs.get('validation_fraction_low'),
                                                          high=0.4 if self.kwargs.get('validation_fraction_high') is None else self.kwargs.get('validation_fraction_high')
                                                          ),
                    n_iter_no_change=np.random.randint(low=2 if self.kwargs.get('n_iter_no_change_low') is None else self.kwargs.get('n_iter_no_change_low'),
                                                       high=10 if self.kwargs.get('n_iter_no_change_high') is None else self.kwargs.get('n_iter_no_change_high')
                                                       ),
                    alpha=np.random.uniform(low=0.01 if self.kwargs.get('alpha_low') is None else self.kwargs.get('alpha_low'),
                                            high=0.99 if self.kwargs.get('alpha_high') is None else self.kwargs.get('alpha_high')
                                            ),
                    ccp_alpha=np.random.uniform(low=0.0 if self.kwargs.get('ccp_alpha_low') is None else self.kwargs.get('ccp_alpha_low'),
                                                high=1.0 if self.kwargs.get('ccp_alpha_high') is None else self.kwargs.get('ccp_alpha_high')
                                                )
                    )

    def k_nearest_neighbor(self) -> KNeighborsRegressor:
        """
        Config K-Nearest-Neighbor (KNN) Regressor

        :return KNeighborsRegressors
            Model object
        """
        return KNeighborsRegressor(n_neighbors=self.reg_params.get('n_neighbors'),
                                   weights=self.reg_params.get('weights'),
                                   algorithm=self.reg_params.get('algorithm'),
                                   leaf_size=self.reg_params.get('leaf_size'),
                                   p=self.reg_params.get('p'),
                                   n_jobs=os.cpu_count()
                                   )

    def k_nearest_neighbor_param(self) -> dict:
        """
        Generate K-Nearest Neighbor regressor parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(n_neighbors=np.random.randint(low=2 if self.kwargs.get('n_neighbors_low') is None else self.kwargs.get('n_neighbors_low'),
                                                  high=12 if self.kwargs.get('n_neighbors_high') is None else self.kwargs.get('n_neighbors_high')
                                                  ),
                    weights=np.random.choice(a=['uniform', 'distance'] if self.kwargs.get('weights_choice') is None else self.kwargs.get('weights_choice')),
                    algorithm=np.random.choice(a=['auto', 'ball_tree', 'kd_tree', 'brute'] if self.kwargs.get('algorithm_choice') is None else self.kwargs.get('algorithm_choice')),
                    leaf_size=np.random.randint(low=15 if self.kwargs.get('leaf_size_low') is None else self.kwargs.get('leaf_size_low'),
                                                high=100 if self.kwargs.get('leaf_size_high') is None else self.kwargs.get('leaf_size_high')
                                                ),
                    p=np.random.choice(a=[1, 2, 3] if self.kwargs.get('p_choice') is None else self.kwargs.get('p_choice')),
                    #metric=np.random.choice(a=['minkowski', 'precomputed'])
                    )

    def lasso_regression(self) -> Lasso:
        """
        Config Lasso Regression

        :return: Lasso
            Model object
        """
        return Lasso(alpha=self.reg_params.get('alpha'),
                     fit_intercept=self.reg_params.get('fit_intercept'),
                     precompute=self.reg_params.get('precompute'),
                     max_iter=self.reg_params.get('max_iter'),
                     random_state=self.seed,
                     selection=self.reg_params.get('selection')
                     )

    def lasso_regression_param(self) -> dict:
        """
        Generate Lasso Regression parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(alpha=np.random.uniform(low=0.0 if self.kwargs.get('alpha_low') is None else self.kwargs.get('alpha_low'),
                                            high=1.0 if self.kwargs.get('alpha_high') is None else self.kwargs.get('alpha_high')
                                            ),
                    precompute=np.random.choice(a=[True, False] if self.kwargs.get('precompute_choice') is None else self.kwargs.get('precompute_choice')),
                    max_iter=np.random.randint(low=5 if self.kwargs.get('max_iter_low') is None else self.kwargs.get('max_iter_low'),
                                               high=1000 if self.kwargs.get('max_iter_high') is None else self.kwargs.get('max_iter_high')
                                               ),
                    fit_intercept=np.random.choice(a=[True, False] if self.kwargs.get('fit_intercept_choice') is None else self.kwargs.get('fit_intercept_choice')),
                    selection=np.random.choice(a=['cyclic', 'random'] if self.kwargs.get('selection_choice') is None else self.kwargs.get('selection_choice'))
                    )

    def random_forest(self) -> RandomForestRegressor:
        """
        Config Random Forest Regressor

        :return: RandomForestRegressor
            Model object
        """
        return RandomForestRegressor(n_estimators=self.reg_params.get('n_estimators'),
                                     criterion=self.reg_params.get('criterion'),
                                     max_depth=self.reg_params.get('max_depth'),
                                     min_samples_split=self.reg_params.get('min_samples_split'),
                                     min_samples_leaf=self.reg_params.get('min_samples_leaf'),
                                     bootstrap=self.reg_params.get('bootstrap'),
                                     n_jobs=os.cpu_count(),
                                     random_state=self.seed
                                     )

    def random_forest_param(self) -> dict:
        """
        Generate Random Forest regressor parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(n_estimators=np.random.randint(low=5 if self.kwargs.get('n_estimators_low') is None else self.kwargs.get('n_estimators_low'),
                                                   high=100 if self.kwargs.get('n_estimators_high') is None else self.kwargs.get('n_estimators_high')
                                                   ),
                    criterion=np.random.choice(a=['mse', 'mae'] if self.kwargs.get('criterion_choice') is None else self.kwargs.get('criterion_choice')),
                    max_depth=np.random.randint(low=1 if self.kwargs.get('max_depth_low') is None else self.kwargs.get('max_depth_low'),
                                                high=12 if self.kwargs.get('max_depth_high') is None else self.kwargs.get('max_depth_high')
                                                ),
                    min_samples_split=np.random.randint(low=2 if self.kwargs.get('min_samples_split_low') is None else self.kwargs.get('min_samples_split_low'),
                                                        high=6 if self.kwargs.get('min_samples_split_high') is None else self.kwargs.get('min_samples_split_high')
                                                        ),
                    min_samples_leaf=np.random.randint(low=1 if self.kwargs.get('min_samples_leaf_low') is None else self.kwargs.get('min_samples_leaf_low'),
                                                       high=6 if self.kwargs.get('min_samples_leaf_high') is None else self.kwargs.get('min_samples_leaf_high')
                                                       ),
                    bootstrap=np.random.choice(a=[True, False] if self.kwargs.get('bootstrap_choice') is None else self.kwargs.get('bootstrap_choice'))
                    )

    def support_vector_machine(self) -> SVR:
        """
        Config of the Support Vector Machine Regressor

        :return: SVR
            Model object
        """
        return SVR(C=self.reg_params.get('C'),
                   kernel=self.reg_params.get('kernel'),
                   degree=self.reg_params.get('degree'),
                   shrinking=self.reg_params.get('shrinking'),
                   cache_size=self.reg_params.get('cache_size'),
                   max_iter=self.reg_params.get('max_iter')
                   )

    def support_vector_machine_param(self) -> dict:
        """
        Generate Support Vector Machine regressor parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(C=np.random.uniform(low=0.0001 if self.kwargs.get('C_low') is None else self.kwargs.get('C_low'),
                                        high=1.0 if self.kwargs.get('C_high') is None else self.kwargs.get('C_high')
                                        ),
                    kernel=np.random.choice(a=['rbf', 'linear', 'poly', 'sigmoid'] if self.kwargs.get('kernel_choice') is None else self.kwargs.get('kernel_choice')), #'precomputed'
                    #gamma=np.random.choice(a=['auto', 'scale']),
                    shrinking=np.random.choice(a=[True, False] if self.kwargs.get('shrinking_choice') is None else self.kwargs.get('shrinking_choice')),
                    cache_size=np.random.randint(low=100 if self.kwargs.get('cache_size_low') is None else self.kwargs.get('cache_size_low'),
                                                 high=500 if self.kwargs.get('cache_size_high') is None else self.kwargs.get('cache_size_high')
                                                 ),
                    decision_function_shape=np.random.choice(a=['ovo', 'ovr'] if self.kwargs.get('decision_function_shape_choice') is None else self.kwargs.get('decision_function_shape_choice')),
                    max_iter=np.random.randint(low=10 if self.kwargs.get('max_iter_low') is None else self.kwargs.get('max_iter_low'),
                                               high=100 if self.kwargs.get('max_iter_high') is None else self.kwargs.get('max_iter_high')
                                               )
                    )

    def linear_support_vector_machine(self) -> LinearSVR:
        """
        Config of the Support Vector Machine Regressor

        :return: LinearSVR
            Model object
        """
        return LinearSVR(C=self.reg_params.get('C'),
                         loss=self.reg_params.get('loss'),
                         random_state=self.seed,
                         max_iter=self.reg_params.get('max_iter')
                         )

    def linear_support_vector_machine_param(self) -> dict:
        """
        Generate Linear Support Vector Machine regressor parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(C=np.random.uniform(low=0.0001 if self.kwargs.get('C_low') is None else self.kwargs.get('C_low'),
                                        high=1.0 if self.kwargs.get('C_high') is None else self.kwargs.get('C_high')
                                        ),
                    penalty=np.random.choice(a=['l1', 'l2'] if self.kwargs.get('penalty_choice') is None else self.kwargs.get('penalty_choice')),
                    loss=np.random.choice(a=['hinge', 'squared_hinge'] if self.kwargs.get('loss_choice') is None else self.kwargs.get('loss_choice')),
                    max_iter=np.random.randint(low=10 if self.kwargs.get('max_iter_low') is None else self.kwargs.get('max_iter_low'),
                                               high=100 if self.kwargs.get('max_iter_high') is None else self.kwargs.get('max_iter_high')
                                               )
                    )

    def nu_support_vector_machine(self) -> NuSVR:
        """
        Config of the Support Vector Machine Regressor

        :return: SVR
            Model object
        """
        return NuSVR(nu=self.reg_params.get('nu'),
                     C=self.reg_params.get('C'),
                     kernel=self.reg_params.get('kernel'),
                     degree=self.reg_params.get('degree'),
                     shrinking=self.reg_params.get('shrinking'),
                     cache_size=self.reg_params.get('cache_size'),
                     max_iter=self.reg_params.get('max_iter')
                     )

    def nu_support_vector_machine_param(self) -> dict:
        """
        Generate Nu-Support Vector Machine regressor parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(C=np.random.uniform(low=0.0001 if self.kwargs.get('C_low') is None else self.kwargs.get('C_low'),
                                        high=1.0 if self.kwargs.get('C_high') is None else self.kwargs.get('C_high')
                                        ),
                    nu=np.random.uniform(low=0.01 if self.kwargs.get('nu_low') is None else self.kwargs.get('nu_low'),
                                         high=0.99 if self.kwargs.get('nu_high') is None else self.kwargs.get('nu_high')
                                         ),
                    kernel=np.random.choice(a=['rbf', 'linear', 'poly', 'sigmoid'] if self.kwargs.get('kernel_choice') is None else self.kwargs.get('kernel_choice')), #'precomputed'
                    #gamma=np.random.choice(a=['auto', 'scale']),
                    shrinking=np.random.choice(a=[True, False] if self.kwargs.get('shrinking_choice') is None else self.kwargs.get('shrinking_choice')),
                    cache_size=np.random.randint(low=100 if self.kwargs.get('cache_size_low') is None else self.kwargs.get('cache_size_low'),
                                                 high=500 if self.kwargs.get('cache_size_high') is None else self.kwargs.get('cache_size_high')
                                                 ),
                    decision_function_shape=np.random.choice(a=['ovo', 'ovr'] if self.kwargs.get('decision_function_shape_choice') is None else self.kwargs.get('decision_function_shape_choice')),
                    max_iter=np.random.randint(low=10 if self.kwargs.get('max_iter_low') is None else self.kwargs.get('max_iter_low'),
                                               high=100 if self.kwargs.get('max_iter_high') is None else self.kwargs.get('max_iter_high')
                                               )
                    )


class ModelGeneratorClf(Classification):
    """
    Class for generating supervised learning classification models
    """
    def __init__(self,
                 model_name: str = None,
                 clf_params: dict = None,
                 models: List[str] = None,
                 labels: List[str] = None,
                 model_id: int = 0,
                 seed: int = 1234,
                 **kwargs
                 ):
        """
        :param model_name: str
            Abrreviate name of the model

        :param clf_params: dict
            Pre-configured classification model parameter

        :param models: List[str]
            Names of the possible models to sample from

        :param labels: List[str]
            Class labels

        :param model_id: int
            Model identifier

        :param seed: int
            Seed

        :param kwargs: dict
            Key-word arguments
        """
        super().__init__(clf_params=clf_params, seed=seed, **kwargs)
        self.id: int = model_id
        self.fitness: dict = {}
        self.fitness_score: float = 0.0
        self.models: List[str] = models
        self.model_name: str = model_name
        if self.model_name is None:
            self.random: bool = True
            if self.models is not None:
                for model in self.models:
                    if model not in CLF_ALGORITHMS.keys():
                        self.random: bool = False
                        raise SupervisedMLException(f'Model ({model}) is not supported. Supported classification models are: {list(CLF_ALGORITHMS.keys())}')
        else:
            if self.model_name not in CLF_ALGORITHMS.keys():
                raise SupervisedMLException(f'Model ({self.model_name}) is not supported. Supported classification models are: {list(CLF_ALGORITHMS.keys())}')
            else:
                self.random: bool = False
        self.model = None
        self.model_param: dict = {}
        self.model_param_mutated: dict = {}
        self.model_param_mutation: str = ''
        self.features: List[str] = []
        self.target: str = ''
        self.target_labels: List[str] = labels
        self.train_time = None
        self.multi = None
        self.creation_time: str = None

    def generate_model(self):
        """
        Generate supervised machine learning model with randomized parameter configuration
        """
        if self.random:
            if self.models is None:
                self.model_name = copy.deepcopy(np.random.choice(a=list(CLF_ALGORITHMS.keys())))
            else:
                self.model_name = copy.deepcopy(np.random.choice(a=self.models))
            _model = copy.deepcopy(CLF_ALGORITHMS.get(self.model_name))
        else:
            _model = copy.deepcopy(CLF_ALGORITHMS.get(self.model_name))
        if len(self.clf_params.keys()) == 0:
            self.model_param = getattr(Classification(**self.kwargs), f'{_model}_param')()
            self.clf_params = copy.deepcopy(self.model_param)
            _idx: int = 0 if len(self.model_param_mutated.keys()) == 0 else len(self.model_param_mutated.keys()) + 1
            self.model_param_mutated.update({str(_idx): {copy.deepcopy(self.model_name): {}}})
            for param in self.model_param.keys():
                self.model_param_mutated[str(_idx)][copy.deepcopy(self.model_name)].update({param: copy.deepcopy(self.model_param.get(param))})
        else:
            if len(self.model_param_mutation) > 0:
                self.model_param = getattr(Classification(**self.kwargs), f'{_model}_param')()
                self.clf_params = copy.deepcopy(self.model_param)
            else:
                self.model_param = copy.deepcopy(self.clf_params)
        self.model_param_mutation = 'new_model'
        self.model = copy.deepcopy(getattr(Classification(clf_params=self.clf_params), _model)())
        Log().log(msg=f'Generate classifier: {self.model}')

    def generate_params(self, param_rate: float = 0.1, force_param: dict = None):
        """
        Generate parameter for supervised learning models

        :param param_rate: float
            Rate of parameters of each model to mutate

        :param force_param: dict
            Parameter config to force explicitly
        """
        if param_rate > 1:
            _rate: float = 1.0
        else:
            if param_rate > 0:
                _rate: float = param_rate
            else:
                _rate: float = 0.1
        _params: dict = getattr(Classification(**self.kwargs), f'{CLF_ALGORITHMS.get(self.model_name)}_param')()
        _force_param: dict = {} if force_param is None else force_param
        _param_choices: List[str] = [p for p in _params.keys() if p not in _force_param.keys()]
        _gen_n_params: int = round(len(_params.keys()) * _rate)
        if _gen_n_params == 0:
            _gen_n_params = 1
        elif _gen_n_params > len(_param_choices):
            _gen_n_params = len(_param_choices)
        self.model_param_mutated.update({len(self.model_param_mutated.keys()) + 1: {copy.deepcopy(self.model_name): {}}})
        _new_model_params: dict = copy.deepcopy(self.model_param)
        _already_mutated_params: List[str] = []
        for param in _force_param.keys():
            _new_model_params.update({param: _force_param.get(param)})
        for _ in range(0, _gen_n_params, 1):
            _param: str = np.random.choice(a=_param_choices)
            if _param in _already_mutated_params:
                _counter: int = 0
                while _param not in _already_mutated_params or _counter > 100:
                    _param: str = np.random.choice(a=_param_choices)
                    _counter += 1
            _already_mutated_params.append(_param)
            if _params.get(_param) == _new_model_params.get(_param):
                _counter: int = 0
                _next_attempt: dict = getattr(Classification(**self.kwargs), f'{CLF_ALGORITHMS.get(self.model_name)}_param')()
                while _next_attempt.get(_param) == _new_model_params.get(_param) or _counter > 100:
                    _next_attempt: dict = getattr(Classification(**self.kwargs), f'{CLF_ALGORITHMS.get(self.model_name)}_param')()
                    _counter += 1
                _new_model_params.update({_param: copy.deepcopy(_next_attempt.get(_param))})
            else:
                _new_model_params.update({_param: copy.deepcopy(_params.get(_param))})
            Log().log(msg=f'Change hyperparameter: {_param} from {self.model_param.get(_param)} to {_new_model_params.get(_param)} of model {self.model_name}')
            self.model_param_mutated[list(self.model_param_mutated.keys())[-1]][copy.deepcopy(self.model_name)].update({_param: _params.get(_param)})
        self.model_param_mutation = 'params'
        self.model_param = copy.deepcopy(_new_model_params)
        self.clf_params = self.model_param
        self.model = getattr(Classification(clf_params=self.clf_params, **self.kwargs), CLF_ALGORITHMS.get(self.model_name))()
        Log().log(msg=f'Generate hyperparameter for classifier: Rate={param_rate}, Changed={self.model_param_mutated}')

    def get_standard_model_parameter(self) -> dict:
        """
        Get "standard" parameter config of given classification models

        :return dict:
            Standard parameter config of given classification models
        """
        Log().log(msg=f'Get standard hyperparameter for classifier: {self.model_name}')
        return CLF_STANDARD_PARAM.get(self.model_name)

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
                raise SupervisedMLException(f'Model ({self.model_name}) has no function called "predict_proba"')
        else:
            if hasattr(self.model, 'predict'):
                return self.model.predict(x).flatten()
            else:
                raise SupervisedMLException(f'Model ({self.model_name}) has no function called "predict"')

    def train(self, x: np.ndarray, y: np.array, validation: dict = None):
        """
        Train or fit supervised machine learning model

        :param x: np.ndarray
            Train data set

        :param y: np.array
            Target data set

        :param validation: dict
        """
        Log().log(msg=f'Train classifier: Model={self.model_name}, Cases={x.shape[0]}, Predictors={x.shape[1]}, Hyperparameter={self.model_param}')
        _t0: datetime = datetime.now()
        if hasattr(self.model, 'fit'):
            if 'eval_set' in self.model.fit.__code__.co_varnames and validation is not None:
                if hasattr(self.model, 'fit_transform'):
                    self.model.fit_transform(x, y)
                else:
                    self.model.fit(x,
                                   y,
                                   eval_set=[(validation.get('x_val'), validation.get('y_val'))],
                                   early_stopping_rounds=np.random.randint(low=1, high=15) if self.model_param.get('early_stopping') else None,
                                   verbose=False
                                   )
            else:
                if hasattr(self.model, 'fit_transform'):
                    self.model.fit_transform(x, y)
                else:
                    self.model.fit(x, y)
        elif hasattr(self.model, 'train'):
            with joblib.parallel_backend(backend='dask'):
                self.model.train(x, y)
        else:
            raise SupervisedMLException('Training (fitting) method not supported by given model object')
        self.train_time = (datetime.now() - _t0).seconds
        self.multi = True if len(pd.unique(values=y)) > 2 else False
        self.creation_time = str(datetime.now())
        Log().log(msg=f'Classifier trained after {self.train_time} seconds')


class ModelGeneratorReg(Regression):
    """
    Class for generating supervised learning regression models
    """
    def __init__(self,
                 model_name: str = None,
                 reg_params: dict = None,
                 models: List[str] = None,
                 cpu_cores: int = 0,
                 seed: int = 1234,
                 **kwargs
                 ):
        """
        :param reg_params: dict
            Pre-configured regression model parameter

        :param models: List[str]
            Names of the possible models to sample from

        :param cpu_cores: int
            Number of CPU core to use

        :param seed: int
            Seed

        :param kwargs: dict
            Key-word arguments
        """
        super().__init__(reg_params=reg_params, cpu_cores=cpu_cores, seed=seed, **kwargs)
        self.id: int = 0
        self.fitness: dict = {}
        self.fitness_score: float = 0.0
        self.models: List[str] = models
        self.model_name: str = model_name
        if self.model_name is not None:
            if self.model_name not in REG_ALGORITHMS.keys():
                raise SupervisedMLException(f'Model ({self.model_name}) is not supported. Supported regression models are: {list(REG_ALGORITHMS.keys())}')
            else:
                self.random: bool = False
        else:
            self.random: bool = True
        self.model = None
        self.model_param: dict = {}
        self.model_param_mutated: dict = {}
        self.model_param_mutation: str = ''
        self.train_time = None
        self.creation_time: str = None

    def generate_model(self):
        """
        Generate supervised machine learning model with randomized parameter configuration
        """
        if self.random:
            if self.models is None:
                self.model_name = copy.deepcopy(np.random.choice(a=list(REG_ALGORITHMS.keys())))
            else:
                self.model_name = copy.deepcopy(np.random.choice(a=self.models))
            _model = copy.deepcopy(REG_ALGORITHMS.get(self.model_name))
        else:
            _model = copy.deepcopy(REG_ALGORITHMS.get(self.model_name))
        if len(self.reg_params.keys()) == 0:
            self.model_param = getattr(Regression(**self.kwargs), f'{_model}_param')()
            self.reg_params = copy.deepcopy(self.model_param)
            _idx: int = 0 if len(self.model_param_mutated.keys()) == 0 else len(self.model_param_mutated.keys()) + 1
            self.model_param_mutated.update({str(_idx): {copy.deepcopy(self.model_name): {}}})
            for param in self.model_param.keys():
                self.model_param_mutated[str(_idx)][copy.deepcopy(self.model_name)].update(
                    {param: copy.deepcopy(self.model_param.get(param))})
        else:
            if len(self.model_param_mutation) > 0:
                self.model_param = getattr(Regression(**self.kwargs), f'{_model}_param')()
                self.reg_params = copy.deepcopy(self.model_param)
            else:
                self.model_param = copy.deepcopy(self.reg_params)
        self.model_param_mutation = 'new_model'
        self.model = getattr(Regression(reg_params=self.reg_params, **self.kwargs), _model)()
        Log().log(msg=f'Generate regressor: {self.model}')

    def generate_params(self, param_rate: float = 0.1, force_param: dict = None):
        """
        Generate parameter for supervised learning models

        :param param_rate: float
            Rate of parameters of each model to mutate

        :param force_param: dict
            Parameter config to force explicitly
        """
        if param_rate > 1:
            _rate: float = 1.0
        else:
            if param_rate > 0:
                _rate: float = param_rate
            else:
                _rate: float = 0.1
        _params: dict = getattr(Regression(**self.kwargs), f'{REG_ALGORITHMS.get(self.model_name)}_param')()
        _force_param: dict = {} if force_param is None else force_param
        _param_choices: List[str] = [p for p in _params.keys() if p not in _force_param.keys()]
        _gen_n_params: int = round(len(_params.keys()) * _rate)
        if _gen_n_params == 0:
            _gen_n_params = 1
        elif _gen_n_params > len(_param_choices):
            _gen_n_params = len(_param_choices)
        self.model_param_mutated.update(
            {len(self.model_param_mutated.keys()) + 1: {copy.deepcopy(self.model_name): {}}})
        _new_model_params: dict = copy.deepcopy(self.model_param)
        _already_mutated_params: List[str] = []
        for param in _force_param.keys():
            _new_model_params.update({param: _force_param.get(param)})
        for _ in range(0, _gen_n_params, 1):
            _param: str = np.random.choice(a=_param_choices)
            if _param in _already_mutated_params:
                _counter: int = 0
                while _param not in _already_mutated_params or _counter > 100:
                    _param: str = np.random.choice(a=_param_choices)
                    _counter += 1
            _already_mutated_params.append(_param)
            if _params.get(_param) == _new_model_params.get(_param):
                _counter: int = 0
                _next_attempt: dict = getattr(Regression(**self.kwargs), f'{REG_ALGORITHMS.get(self.model_name)}_param')()
                while _next_attempt.get(_param) == _new_model_params.get(_param) or _counter > 100:
                    _next_attempt: dict = getattr(Regression(**self.kwargs), f'{REG_ALGORITHMS.get(self.model_name)}_param')()
                    _counter += 1
                _new_model_params.update({_param: copy.deepcopy(_next_attempt.get(_param))})
            else:
                _new_model_params.update({_param: copy.deepcopy(_params.get(_param))})
            Log().log(msg=f'Change hyperparameter: {_param} from {self.model_param.get(_param)} to {_new_model_params.get(_param)} of model {self.model_name}')
            self.model_param_mutated[list(self.model_param_mutated.keys())[-1]][copy.deepcopy(self.model_name)].update(
                {_param: _params.get(_param)})
        self.model_param_mutation = 'params'
        self.model_param = copy.deepcopy(_new_model_params)
        self.reg_params = self.model_param
        self.model = getattr(Regression(reg_params=self.reg_params, **self.kwargs), REG_ALGORITHMS.get(self.model_name))()
        Log().log(msg=f'Generate hyperparameter for regressor: Rate={param_rate}, Changed={self.model_param_mutated}')

    def get_standard_model_parameter(self) -> dict:
        """
        Get parameter "standard" config of given regression models

        :return dict:
            Standard parameter config of given regression models
        """
        Log().log(msg=f'Get standard hyperparameter for regressor: {self.model_name}')
        return REG_STANDARD_PARAM.get(self.model_name)

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
            raise SupervisedMLException(f'Model ({self.model_name}) has no function called "predict"')

    def train(self, x: np.ndarray, y: np.array, validation: dict = None):
        """
        Train or fit supervised machine learning model

        :param x: np.ndarray
            Train data set

        :param y: np.array
            Target data set

        :param validation: dict
        """
        Log().log(msg=f'Train regressor: Model={self.model_name}, Cases={x.shape[0]}, Predictors={x.shape[1]}, Hyperparameter={self.model_param}')
        _t0: datetime = datetime.now()
        if hasattr(self.model, 'fit'):
            if 'eval_set' in self.model.fit.__code__.co_varnames and validation is not None:
                self.model.fit(x,
                               y,
                               eval_set=[(validation.get('x_val'), validation.get('y_val'))],
                               early_stopping_rounds=np.random.randint(low=1, high=15) if self.model_param.get('early_stopping') else None,
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
            raise SupervisedMLException('Training (fitting) method not supported by given model object')
        self.train_time = (datetime.now() - _t0).seconds
        self.creation_time: str = str(datetime.now())
        Log().log(msg=f'Regressor trained after {self.train_time} seconds')
