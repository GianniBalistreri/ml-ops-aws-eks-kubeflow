"""

Task: ... (Function to run in container)

"""

import argparse
import ast
import numpy as np
import os
import pandas as pd
import warnings

from aws import load_file_from_s3_as_df, save_file_to_s3
from custom_logger import Log
from evaluate_machine_learning import EvalClf, EvalReg, ML_METRIC, sml_fitness_score, SML_SCORE
from file_handler import file_handler
from typing import List, NamedTuple

warnings.filterwarnings("ignore", category=FutureWarning)

PARSER = argparse.ArgumentParser(description="calculate metrics for evaluating machine learning models")
PARSER.add_argument('-ml_type', type=str, required=True, default=None, help='machine learning type')
PARSER.add_argument('-target_feature_name', type=str, required=True, default=None, help='name of the target feature')
PARSER.add_argument('-prediction_feature_name', type=str, required=True, default=None, help='name of the prediction variable')
PARSER.add_argument('-train_data_set_path', type=str, required=True, default=None, help='complete file path of the training data set')
PARSER.add_argument('-test_data_set_path', type=str, required=True, default=None, help='complete file path of the test data set')
PARSER.add_argument('-metrics', nargs='+', required=False, default=None, help='metrics to calculate')
PARSER.add_argument('-labels', nargs='+', required=False, default=None, help='class labels')
PARSER.add_argument('-model_id', type=str, required=False, default=None, help='model identifier')
PARSER.add_argument('-sep', type=str, required=False, default=',', help='column separator')
PARSER.add_argument('-val_data_set_path', type=str, required=False, default=None, help='complete file path of the validation data set')
PARSER.add_argument('-output_path_metadata', type=str, required=True, default=None, help='file path of the metadata output')
PARSER.add_argument('-output_path_r2_metric', type=str, required=True, default=None, help='file path of the r2 test metric output')
PARSER.add_argument('-output_path_rmse_norm_metric', type=str, required=True, default=None, help='file path of the normalized rmse test metric output')
PARSER.add_argument('-output_path_accuracy_metric', type=str, required=True, default=None, help='file path of the accuracy test metric output')
PARSER.add_argument('-output_path_precision_metric', type=str, required=True, default=None, help='file path of the precision test metric output')
PARSER.add_argument('-output_path_recall_metric', type=str, required=True, default=None, help='file path of the recall test metric output')
PARSER.add_argument('-output_path_f1_metric', type=str, required=True, default=None, help='file path of the f1 test metric output')
PARSER.add_argument('-output_path_mcc_metric', type=str, required=True, default=None, help='file path of the mcc test metric output')
PARSER.add_argument('-output_path_roc_auc_metric', type=str, required=True, default=None, help='file path of the roc auc test metric output')
PARSER.add_argument('-output_path_cohen_kappa_metric', type=str, required=True, default=None, help='file path of the cohens kappa test metric output')
PARSER.add_argument('-output_path_sml_score_metric', type=str, required=True, default=None, help='file path of the sml test metric output')
PARSER.add_argument('-s3_output_path_metrics', type=str, required=True, default=None, help='complete file path of the metrics output')
PARSER.add_argument('-s3_output_path_confusion_matrix', type=str, required=False, default=None, help='complete file path of the confusion matrix putput')
PARSER.add_argument('-s3_output_path_roc_curve', type=str, required=False, default=None, help='complete file path of the roc curve putput')
PARSER.add_argument('-s3_output_path_metric_table', type=str, required=False, default=None, help='complete file path of the metric table putput')
PARSER.add_argument('-s3_output_path_visualization', type=str, required=False, default=None, help='complete file path of the visualization putput')
ARGS = PARSER.parse_args()


class ModelEvaluationException(Exception):
    """
    Class for handling exceptions for function evaluate_machine_learning
    """
    pass


def _generate_standard_visualizations(obs: np.ndarray,
                                      pred: np.ndarray,
                                      ml_type: str,
                                      file_path: str,
                                      target_labels: List[str] = None
                                      ) -> dict:
    """
    Generate standard visualization based on the machine learning type

    :param obs: np.ndarray
        Observations

    :param pred: np.ndarray
        Predictions

    :param ml_type: str
        Abbreviated name of the machine learning type

    :param file_path: str
        Path of the visualizations to save

    :param target_labels: List[str]
        Labels of the target feature (classification only)

    :return: dict
        Subplot configuration
    """
    _plot: dict = {}
    _best_model_results: pd.DataFrame = pd.DataFrame(data=dict(obs=obs, pred=pred))
    if ml_type == 'reg':
        _best_model_results['abs_diff'] = _best_model_results['obs'] - _best_model_results['pred']
        _best_model_results['rel_diff'] = _best_model_results['obs'] / _best_model_results['pred']
    elif ml_type == 'clf_multi':
        _best_model_results['abs_diff'] = _best_model_results['obs'] - _best_model_results['pred']
    _best_model_results = _best_model_results.round(decimals=4)
    if ml_type == 'reg':
        _plot.update({'Prediction Evaluation of final inherited ML Model:': dict(df=_best_model_results.to_dict(),
                                                                                 features=['obs', 'abs_diff', 'rel_diff', 'pred'],
                                                                                 color_feature='pred',
                                                                                 plot_type='parcoords',
                                                                                 file_path=os.path.join(file_path, 'prediction_evaluation_coords.html'),
                                                                                 kwargs=dict(layout={})
                                                                                 ),
                      'Prediction vs. Observation of final inherited ML Model:': dict(df=_best_model_results.to_dict(),
                                                                                      features=['obs', 'pred'],
                                                                                      plot_type='joint',
                                                                                      file_path=os.path.join(file_path, 'prediction_scatter_contour.html'),
                                                                                      kwargs=dict(layout={})
                                                                                      )
                      })
    else:
        _confusion_matrix: pd.DataFrame = pd.DataFrame(data=EvalClf(obs=obs, pred=pred).confusion(),
                                                       index=target_labels,
                                                       columns=target_labels
                                                       )
        _cf_row_sum = pd.DataFrame()
        _cf_row_sum[' '] = _confusion_matrix.sum()
        _confusion_matrix = pd.concat([_confusion_matrix, _cf_row_sum.transpose()], axis=0)
        _cf_col_sum = pd.DataFrame()
        _cf_col_sum[' '] = _confusion_matrix.transpose().sum()
        _confusion_matrix = pd.concat([_confusion_matrix, _cf_col_sum], axis=1)
        _plot.update({'Confusion Matrix:': dict(df=_confusion_matrix.to_dict(),
                                                features=_confusion_matrix.columns.to_list(),
                                                plot_type='table',
                                                file_path=os.path.join(file_path, 'prediction_confusion_table.html'),
                                                kwargs=dict(layout={})
                                                ),
                      #'Confusion Matrix Heatmap:': dict(df=_best_model_results.to_dict(),
                      #                                  features=_best_model_results.columns.to_list(),
                      #                                  plot_type='heat',
                      #                                  file_path=os.path.join(file_path, 'prediction_confusion_heatmap.html'),
                      #                                  kwargs=dict(layout={})
                      #                                  )
                      })
        #_confusion_matrix_normalized: pd.DataFrame = pd.DataFrame(data=EvalClf(obs=obs, pred=pred).confusion(normalize='pred'),
        #                                                          index=target_labels,
        #                                                          columns=target_labels
        #                                                          )
        #_plot.update({'Confusion Matrix Normalized Heatmap:': dict(df=_confusion_matrix_normalized.to_dict(),
        #                                                           features=_confusion_matrix_normalized.columns.to_list(),
        #                                                           plot_type='heat',
        #                                                           file_path=os.path.join(file_path, 'prediction_confusion_normal_heatmap.html'),
        #                                                           kwargs=dict(layout={})
        #                                                           )
        #              })
        #_plot.update({'Classification Report:': dict(df=_best_model_results.to_dict(),
        #                                             features=_best_model_results.columns.to_list(),
        #                                             plot_type='table',
        #                                             file_path=os.path.join(file_path, 'prediction_clf_report_table.html'),
        #                                             kwargs=dict(layout={})
        #                                             )
        #              })
        if ml_type == 'clf_multi':
            _plot.update({'Prediction Evaluation of final inherited ML Model:': dict(df=_best_model_results.to_dict(),
                                                                                     features=['obs', 'abs_diff', 'pred'],
                                                                                     color_feature='pred',
                                                                                     plot_type='parcoords',
                                                                                     file_path=os.path.join(file_path, 'prediction_evaluation_category.html'),
                                                                                     kwargs=dict(layout={})
                                                                                     )
                          })
        else:
            _roc_curve = pd.DataFrame()
            _roc_curve_values: dict = EvalClf(obs=_best_model_results['obs'],
                                              pred=_best_model_results['pred']
                                              ).roc_curve()
            _roc_curve['roc_curve'] = _roc_curve_values['true_positive_rate'][1]
            _roc_curve['baseline'] = _roc_curve_values['false_positive_rate'][1]
            _plot.update({'ROC-AUC Curve:': dict(df=_roc_curve.to_dict(),
                                                 features=['roc_curve', 'baseline'],
                                                 time_features=['baseline'],
                                                 plot_type='line',
                                                 melt=True,
                                                 use_auto_extensions=False,
                                                 xaxis_label=['False Positive Rate'],
                                                 yaxis_label=['True Positive Rate'],
                                                 file_path=os.path.join(file_path, 'prediction_roc_auc_curve.html'),
                                                 kwargs=dict(layout={})
                                                 )
                          })
    return _plot


def evaluate_machine_learning(ml_type: str,
                              target_feature_name: str,
                              prediction_feature_name: str,
                              train_data_set_path: str,
                              test_data_set_path: str,
                              output_path_metadata: str,
                              output_path_r2_metric: str,
                              output_path_rmse_norm_metric: str,
                              output_path_accuracy_metric: str,
                              output_path_precision_metric: str,
                              output_path_recall_metric: str,
                              output_path_f1_metric: str,
                              output_path_mcc_metric: str,
                              output_path_roc_auc_metric: str,
                              output_path_cohen_kappa_metric: str,
                              output_path_sml_score_metric: str,
                              s3_output_path_metrics: str,
                              s3_output_path_visualization: str = None,
                              s3_output_path_confusion_matrix: str = None,
                              s3_output_path_roc_curve: str = None,
                              s3_output_path_metric_table: str = None,
                              metrics: List[str] = None,
                              labels: List[str] = None,
                              model_id: str = None,
                              val_data_set_path: str = None,
                              sep: str = ',',
                              ) -> NamedTuple('outputs', [('metadata', dict),
                                                          ('r2', float),
                                                          ('rmse_norm', float),
                                                          ('accuracy', float),
                                                          ('precision', float),
                                                          ('recall', float),
                                                          ('f1', float),
                                                          ('mcc', float),
                                                          ('roc_auc', float),
                                                          ('cohen_kappa', float),
                                                          ('sml_score', float)
                                                          ]
                                              ):
    """
    Evaluate supervised machine learning models using most common metrics

    :param ml_type: str
        Name of the machine learning problem
            -> reg: Regression
            -> clf_binary: Binary Classification
            -> clf_multi: Multi-Classification
            -> cluster: Clustering

    :param target_feature_name: str
        Name of the target feature

    :param prediction_feature_name: str
        Name of the variable containing predictions

    :param train_data_set_path: str
        Complete file path of the training data set

    :param test_data_set_path: str
        Complete file path of the tests data set

    :param output_path_metadata: str
        File path of the metadata output file

    :param output_path_r2_metric: str
        File path of the r2 test metric output file

    :param output_path_rmse_norm_metric: str
        File path of the normalized rmse test metric output file

    :param output_path_accuracy_metric: str
        Path of the output accuracy test metric results to save

    :param output_path_precision_metric: str
        Path of the output precision test metric results to save

    :param output_path_recall_metric: str
        Path of the output recall test metric results to save

    :param output_path_f1_metric: str
        Path of the output f1 test metric results to save

    :param output_path_mcc_metric: str
        Path of the output mcc test metric results to save

    :param output_path_roc_auc_metric: str
        Path of the output roc auc test metric results to save

    :param output_path_cohen_kappa_metric: str
        Path of the output cohen's kappa test metric results to save

    :param output_path_sml_score_metric: str
        Path of the output sml test metric results to save

    :param s3_output_path_metrics: str
        Complete file path of the metrics output

    :param s3_output_path_visualization: str
        Complete file path of the visualization output

    :param s3_output_path_confusion_matrix: str
        Complete file path of the confusion matrix output

    :param s3_output_path_roc_curve: str
        Complete file path of the roc curve output

    :param s3_output_path_metric_table: str
        Complete file path of the metrics table output

    :param metrics: List[str]
        Abbreviated names of metrics to apply

    :param labels: List[str]
        Class labels of the target feature

    :param model_id: str
        Model identifier

    :param val_data_set_path: str
        Complete file path of the validation data set

    :param sep: str
        Separator

    :return: NamedTuple
        Train and test metric results
    """
    _metric_output_paths: dict = dict(r2=output_path_r2_metric,
                                      rmse_norm=output_path_rmse_norm_metric,
                                      accuracy=output_path_accuracy_metric,
                                      precision=output_path_precision_metric,
                                      recall=output_path_recall_metric,
                                      f1=output_path_f1_metric,
                                      mcc=output_path_mcc_metric,
                                      roc_auc=output_path_roc_auc_metric,
                                      cohen_kappa=output_path_cohen_kappa_metric
                                      )
    _train_df: pd.DataFrame = load_file_from_s3_as_df(file_path=train_data_set_path, sep=sep)
    Log().log(msg=f'Load training data set: {train_data_set_path} -> Cases={_train_df.shape[0]}, Features={_train_df.shape[1]}')
    _test_df: pd.DataFrame = load_file_from_s3_as_df(file_path=test_data_set_path, sep=sep)
    Log().log(msg=f'Load test data set: {test_data_set_path} -> Cases={_test_df.shape[0]}, Features={_test_df.shape[1]}')
    if val_data_set_path is None:
        _val_df: pd.DataFrame = None
    else:
        _val_df: pd.DataFrame = load_file_from_s3_as_df(file_path=val_data_set_path, sep=sep)
        Log().log(msg=f'Load validation data set: {val_data_set_path} -> Cases={_val_df.shape[0]}, Features={_val_df.shape[1]}')
    if metrics is None:
        _metrics: List[str] = ML_METRIC.get(ml_type)
    else:
        _metrics: List[str] = []
        for ml_metric in metrics:
            if ml_metric in ML_METRIC.get(ml_type):
                _metrics.append(ml_metric)
            else:
                Log().log(msg=f'Evaluation metric ({ml_metric}) not supported')
        if len(_metrics) == 0:
            raise ModelEvaluationException('No supported evaluation metric found')
    _evaluation: dict = dict(train={}, test={}, val={}, sml_score=None, model_id=model_id, target=target_feature_name)
    for metric in _metrics:
        if ml_type == 'reg':
            _evaluation['train'].update({metric: getattr(EvalReg(obs=_train_df[target_feature_name].values,
                                                                 pred=_train_df[prediction_feature_name].values
                                                                 ),
                                                         metric)()
                                         })
            Log().log(msg=f'Regression metric (training): {metric}={_evaluation["train"][metric]}')
            _evaluation['test'].update({metric: getattr(EvalReg(obs=_test_df[target_feature_name].values,
                                                                pred=_test_df[prediction_feature_name].values
                                                                ),
                                                        metric)()
                                        })
            Log().log(msg=f'Regression metric (test): {metric}={_evaluation["test"][metric]}')
            if _val_df is not None:
                _evaluation['val'].update({metric: getattr(EvalReg(obs=_val_df[target_feature_name].values,
                                                                   pred=_val_df[prediction_feature_name].values
                                                                   ),
                                                           metric)()
                                           })
                Log().log(msg=f'Regression metric (validation): {metric}={_evaluation["val"][metric]}')
            if metric in _metric_output_paths.keys():
                file_handler(file_path=_metric_output_paths.get(metric), obj=_evaluation['test'][metric])
        else:
            _evaluation['train'].update({metric: getattr(EvalClf(obs=_train_df[target_feature_name].values,
                                                                 pred=_train_df[prediction_feature_name].values
                                                                 ),
                                                         metric)()
                                         })
            Log().log(msg=f'Classification ({ml_type}) metric (training): {metric}={_evaluation["train"][metric]}')
            _evaluation['test'].update({metric: getattr(EvalClf(obs=_test_df[target_feature_name].values,
                                                                pred=_test_df[prediction_feature_name].values
                                                                ),
                                                        metric)()
                                        })
            Log().log(msg=f'Classification ({ml_type}) metric (test): {metric}={_evaluation["test"][metric]}')
            if metric in _metric_output_paths.keys():
                file_handler(file_path=_metric_output_paths.get(metric), obj=_evaluation['test'][metric])
            if metric == 'confusion':
                _df_count: pd.DataFrame = _test_df.groupby([target_feature_name, prediction_feature_name]).size().reset_index(name='count')
                _df_count.to_csv(path_or_buf=s3_output_path_confusion_matrix, sep=sep, header=False, index=False)
                Log().log(msg=f'Save confusion matrix: {s3_output_path_confusion_matrix}')
            if ml_type == 'clf_binary':
                if metric == 'accuracy':
                    _metric_type: str = 'accuracy-score'
                elif metric == 'roc_auc':
                    _metric_type: str = 'roc-auc-score'
                    _df_roc: pd.DataFrame = pd.DataFrame()
                    _roc_curve: dict = EvalClf(obs=_test_df[target_feature_name].values,
                                               pred=_test_df[prediction_feature_name].values,
                                               ).roc_curve()
                    _tpr: List[float] = []
                    _fpr: List[float] = []
                    _thresholds: List[float] = []
                    for i in range(0, len(_test_df[target_feature_name].unique()), 1):
                        _tpr.append(_roc_curve['true_positive_rate'][i])
                        _fpr.append(_roc_curve['false_positive_rate'][i])
                        _thresholds.append(_roc_curve['roc_auc'][i])
                    _df_roc['tpr'] = _tpr
                    _df_roc['fpr'] = _fpr
                    _df_roc['thresholds'] = _thresholds
                    _df_roc.to_csv(path_or_buf=s3_output_path_roc_curve, sep=sep, header=False, index=False)
                    Log().log(msg=f'Save roc-auc score: {s3_output_path_roc_curve}')
            if _val_df is not None:
                _evaluation['val'].update({metric: getattr(EvalClf(obs=_val_df[target_feature_name].values,
                                                                   pred=_val_df[prediction_feature_name].values
                                                                   ),
                                                           metric)()
                                           })
                Log().log(msg=f'Classification ({ml_type}) metric (validation): {metric}={_evaluation["val"][metric]}')
    if SML_SCORE['ml_metric'][ml_type] in list(_evaluation['test'].keys()):
        _evaluation['sml_score'] = sml_fitness_score(ml_metric=tuple([SML_SCORE['ml_metric_best'][ml_type], _evaluation['test'][SML_SCORE['ml_metric'][ml_type]]]),
                                                     train_test_metric=tuple([_evaluation['train'][SML_SCORE['ml_metric'][ml_type]], _evaluation['test'][SML_SCORE['ml_metric'][ml_type]]]),
                                                     train_time_in_seconds=1.0
                                                     )
        file_handler(file_path=output_path_sml_score_metric, obj=_evaluation['sml_score'])
        Log().log(msg=f'Supervised machine learning metric: sml_score={_evaluation["sml_score"]}')
    _df_table = pd.DataFrame(_evaluation)
    _df_table.to_csv(path_or_buf=s3_output_path_metrics, sep=sep)
    Log().log(msg=f'Save metrics: {s3_output_path_metrics}')
    if s3_output_path_visualization is not None:
        _file_name: str = s3_output_path_visualization.split('/')[-1]
        _file_path: str = s3_output_path_visualization.replace(_file_name, '')
        _standard_visualizations: dict = _generate_standard_visualizations(obs=_test_df[target_feature_name].values,
                                                                           pred=_test_df[prediction_feature_name].values,
                                                                           ml_type=ml_type,
                                                                           file_path=_file_path,
                                                                           target_labels=None
                                                                           )
        save_file_to_s3(file_path=s3_output_path_visualization, obj=_standard_visualizations)
        Log().log(msg=f'Save visualization data: {s3_output_path_visualization}')
    _skip_metric_summary_table: List[str] = ['classification_report', 'confusion']
    _df_table: pd.DataFrame = pd.DataFrame()
    _var_data_set_type: List[str] = []
    _var_metric_name: List[str] = []
    _var_metric_value: List[float] = []
    for train_metric_name, train_metric_value in _evaluation['train'].items():
        if train_metric_name not in _skip_metric_summary_table:
            _var_data_set_type.append('train')
            _var_metric_name.append(train_metric_name)
            _var_metric_value.append(train_metric_value)
    for test_metric_name, test_metric_value in _evaluation['test'].items():
        if test_metric_name not in _skip_metric_summary_table:
            _var_data_set_type.append('test')
            _var_metric_name.append(test_metric_name)
            _var_metric_value.append(test_metric_value)
    if val_data_set_path is not None:
        for val_metric_name, val_metric_value in _evaluation['val'].items():
            if val_metric_name not in _skip_metric_summary_table:
                _var_data_set_type.append('val')
                _var_metric_name.append(val_metric_name)
                _var_metric_value.append(val_metric_value)
    _df_table['metric_name'] = _var_metric_name
    _df_table['metric_value'] = _var_metric_value
    _df_table['data_set_type'] = _var_data_set_type
    _df_table.to_csv(path_or_buf=s3_output_path_metric_table, sep=sep, header=False, index=True)
    Log().log(msg=f'Save metric table: {s3_output_path_metric_table}')
    file_handler(file_path=output_path_metadata, obj=_evaluation)
    return [_evaluation,
            _evaluation['test'].get('r2'),
            _evaluation['test'].get('rmse_norm'),
            _evaluation['test'].get('accuracy'),
            _evaluation['test'].get('precision'),
            _evaluation['test'].get('recall'),
            _evaluation['test'].get('f1'),
            _evaluation['test'].get('mcc'),
            _evaluation['test'].get('roc_auc'),
            _evaluation['test'].get('cohen_kappa'),
            _evaluation.get('sml_score'),
            ]


if __name__ == '__main__':
    if ARGS.metrics:
        ARGS.metrics = ast.literal_eval(ARGS.metrics[0])
    if ARGS.labels:
        ARGS.labels = ast.literal_eval(ARGS.labels[0])
    evaluate_machine_learning(ml_type=ARGS.ml_type,
                              target_feature_name=ARGS.target_feature_name,
                              prediction_feature_name=ARGS.prediction_feature_name,
                              train_data_set_path=ARGS.train_data_set_path,
                              test_data_set_path=ARGS.test_data_set_path,
                              output_path_metadata=ARGS.output_path_metadata,
                              output_path_r2_metric=ARGS.output_path_r2_metric,
                              output_path_rmse_norm_metric=ARGS.output_path_rmse_norm_metric,
                              output_path_accuracy_metric=ARGS.output_path_accuracy_metric,
                              output_path_precision_metric=ARGS.output_path_precision_metric,
                              output_path_recall_metric=ARGS.output_path_recall_metric,
                              output_path_f1_metric=ARGS.output_path_f1_metric,
                              output_path_mcc_metric=ARGS.output_path_mcc_metric,
                              output_path_roc_auc_metric=ARGS.output_path_roc_auc_metric,
                              output_path_cohen_kappa_metric=ARGS.output_path_cohen_kappa_metric,
                              output_path_sml_score_metric=ARGS.output_path_sml_score_metric,
                              s3_output_path_metrics=ARGS.s3_output_path_metrics,
                              s3_output_path_visualization=ARGS.s3_output_path_visualization,
                              s3_output_path_confusion_matrix=ARGS.s3_output_path_confusion_matrix,
                              s3_output_path_roc_curve=ARGS.s3_output_path_roc_curve,
                              s3_output_path_metric_table=ARGS.s3_output_path_metric_table,
                              metrics=ARGS.metrics,
                              labels=ARGS.labels,
                              model_id=ARGS.model_id,
                              val_data_set_path=ARGS.val_data_set_path,
                              sep=ARGS.sep
                              )
