"""

Task: ... (Function to run in container)

"""

import argparse
import numpy as np
import os
import pandas as pd

from aws import save_file_to_s3
from custom_logger import Log
from evaluate_machine_learning import EvalClf, EvalReg, ML_METRIC, sml_fitness_score, SML_SCORE
from file_handler import file_handler
from kfp.components import OutputPath
from typing import Dict, List, NamedTuple


PARSER = argparse.ArgumentParser(description="calculate metrics for evaluating machine learning models")
PARSER.add_argument('-ml_type', type=str, required=True, default=None, help='machine learning type')
PARSER.add_argument('-target_feature_name', type=str, required=True, default=None, help='name of the target feature')
PARSER.add_argument('-prediction_feature_name', type=str, required=True, default=None, help='name of the prediction variable')
PARSER.add_argument('-train_data_set_path', type=str, required=True, default=None, help='complete file path of the training data set')
PARSER.add_argument('-test_data_set_path', type=str, required=True, default=None, help='complete file path of the test data set')
PARSER.add_argument('-metrics', type=list, required=False, default=None, help='metrics to calculate')
PARSER.add_argument('-labels', type=list, required=False, default=None, help='class labels')
PARSER.add_argument('-model_id', type=str, required=False, default=None, help='model identifier')
PARSER.add_argument('-sep', type=str, required=False, default=',', help='column separator')
PARSER.add_argument('-val_data_set_path', type=str, required=False, default=None, help='complete file path of the validation data set')
PARSER.add_argument('-output_path_metrics', type=str, required=True, default=None, help='file path of the metric output')
PARSER.add_argument('-metadata_file_path', type=str, required=True, default=None, help='file path of the metadata output')
PARSER.add_argument('-metric_file_path', type=str, required=True, default=None, help='file path of the metric output')
PARSER.add_argument('-table_file_path', type=str, required=True, default=None, help='file path of the table output')
PARSER.add_argument('-output_path_confusion_matrix', type=str, required=True, default=None, help='file path of the confusion matrix putput')
PARSER.add_argument('-output_path_roc_curve', type=str, required=True, default=None, help='file path of the roc curve putput')
PARSER.add_argument('-output_path_confusion_matrix', type=str, required=True, default=None, help='file path of the confusion matrix putput')
PARSER.add_argument('-output_path_metric_table', type=str, required=True, default=None, help='file path of the metric table putput')
PARSER.add_argument('-output_path_metrics_customized', type=str, required=False, default=None, help='complete file path of the metrics output')
ARGS = PARSER.parse_args()

KFP_METRIC_TYPE: List[str] = ['accuracy-score', 'roc-auc-score']


class ModelEvaluationException(Exception):
    """
    Class for handling exceptions for function evaluate_machine_learning
    """
    pass


def _generate_kfp_metric_template(file_paths: List[str],
                                  metric_types: List[str],
                                  metric_values: List[float],
                                  metric_formats: List[str]
                                  ) -> Dict[str, List[Dict[str, str]]]:
    """
    Generate Kubeflow pipeline metric template

    :param file_paths: List[str]
        Complete file path of the plots

    :param metric_types: List[str]
        Name of the pre-defined kfp metrics
            -> accuracy-score: Accuracy
            -> roc-auc-score: ROC-AUC

    :param metric_values: List[float]
        Metric value

    :param metric_formats: List[str]
        Value formats of the Kubeflow pipeline metric
            -> RAW: raw value
            -> PERCENTAGE: scaled percentage value

    :return: Dict[str, List[Dict[str, str]]]
        Configured Kubeflow pipeline metric template
    """
    _metric: Dict[str, List[Dict[str, str]]] = dict(metrics=[])
    for file_path, metric_type, metric_value, metric_format in zip(file_paths, metric_types, metric_values, metric_formats):
        if metric_type in KFP_METRIC_TYPE:
            _plot_config: Dict[str, str] = dict(name=metric_type, format=metric_format, numberValue=metric_values)
        else:
            raise ModelEvaluationException(f'Kubeflow pipeline metric ({metric_type}) not supported')
        _metric['metrics'].append(_plot_config)
    return _metric


def _generate_kfp_visualization_template(file_paths: List[str],
                                         metric_types: List[str],
                                         target_feature: str = None,
                                         prediction_feature: str = None,
                                         labels: List[str] = None,
                                         header: List[str] = None
                                         ) -> Dict[str, List[Dict[str, str]]]:
    """
    Generate Kubeflow pipeline visualization template

    :param file_paths: List[str]
        Complete file path of the plots

    :param metric_types: List[str]
        Name of the pre-defined kfp visualization
            -> confusion_matrix: Confusion matrix
            -> roc: ROC-AUC
            -> table: Data table

    :return: Dict[str, List[Dict[str, str]]]
        Configured Kubeflow pipeline metadata template
    """
    _metadata: Dict[str, List[Dict[str, str]]] = dict(outputs=[])
    for file_path, metric_type in zip(file_paths, metric_types):
        if metric_type == 'confusion_matrix':
            _plot_config: Dict[str, str] = dict(type=metric_type,
                                                format='csv',
                                                schema=None,
                                                target_col=target_feature,
                                                predicted_col=prediction_feature,
                                                labels=labels,
                                                storage='s3',
                                                source=file_path
                                                )
        elif metric_type == 'roc':
            _schema: List[Dict[str, str]] = list()
            _schema.append(dict(name='fpr', type='NUMBER'))
            _schema.append(dict(name='tpr', type='NUMBER'))
            _schema.append(dict(name='thresholds', type='NUMBER'))
            _plot_config: Dict[str, str] = dict(type=metric_type,
                                                format='csv',
                                                schema=_schema,
                                                target_col=target_feature,
                                                predicted_col=prediction_feature,
                                                storage='s3',
                                                source=file_path
                                                )
        elif metric_type == 'table':
            _plot_config: Dict[str, str] = dict(type=metric_type,
                                                format='csv',
                                                header=header,
                                                storage='s3',
                                                source=file_path
                                                )
        else:
            raise ModelEvaluationException(f'Kubeflow pipeline visualization ({metric_type}) not supported')
        _metadata['outputs'].append(_plot_config)
    return _metadata


def generate_standard_visualizations(obs: np.ndarray,
                                     pred: np.ndarray,
                                     ml_type: str,
                                     file_path: str,
                                     target_labels: List[str] = None
                                     ) -> dict:
    """
    Generate standard visualization based on the machine learning type
    :param obs:
    :param pred:
    :param ml_type:
    :param file_path:
    :param target_labels: List[str]
    :return:
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
        _plot.update({'Prediction Evaluation of final inherited ML Model:': dict(df=_best_model_results,
                                                                                 features=['obs', 'abs_diff', 'rel_diff', 'pred'],
                                                                                 color_feature='pred',
                                                                                 plot_type='parcoords',
                                                                                 file_path=os.path.join(file_path, 'ga_prediction_evaluation_coords.html'),
                                                                                 ),
                      'Prediction vs. Observation of final inherited ML Model:': dict(df=_best_model_results,
                                                                                      features=['obs', 'pred'],
                                                                                      plot_type='joint',
                                                                                      file_path=os.path.join(file_path, 'ga_prediction_scatter_contour.html'),
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
        _plot.update({'Confusion Matrix:': dict(df=_confusion_matrix,
                                                features=_confusion_matrix.columns.to_list(),
                                                plot_type='table',
                                                file_path=os.path.join(file_path, 'ga_prediction_confusion_table.html'),
                                                ),
                      'Confusion Matrix Heatmap:': dict(df=_best_model_results,
                                                        features=_best_model_results.columns.to_list(),
                                                        plot_type='heat',
                                                        file_path=os.path.join(file_path, 'ga_prediction_confusion_heatmap.html'),
                                                        )
                      })
        _confusion_matrix_normalized: pd.DataFrame = pd.DataFrame(data=EvalClf(obs=obs, pred=pred).confusion(normalize='pred'),
                                                                  index=target_labels,
                                                                  columns=target_labels
                                                                  )
        _plot.update({'Confusion Matrix Normalized Heatmap:': dict(df=_confusion_matrix_normalized,
                                                                   features=_confusion_matrix_normalized.columns.to_list(),
                                                                   plot_type='heat',
                                                                   file_path=os.path.join(file_path, 'ga_prediction_confusion_normal_heatmap.html'),
                                                                   )
                      })
        _plot.update({'Classification Report:': dict(df=_best_model_results,
                                                     features=_best_model_results.columns.to_list(),
                                                     plot_type='table',
                                                     file_path=os.path.join(file_path, 'ga_prediction_clf_report_table.html'),
                                                     )
                      })
        if ml_type == 'clf_multi':
            _plot.update({'Prediction Evaluation of final inherited ML Model:': dict(df=_best_model_results,
                                                                                     features=['obs', 'abs_diff', 'pred'],
                                                                                     color_feature='pred',
                                                                                     plot_type='parcoords',
                                                                                     brushing=True,
                                                                                     file_path=os.path.join(file_path, 'ga_prediction_evaluation_category.html'),
                                                                                     )
                          })
        else:
            _roc_curve = pd.DataFrame()
            _roc_curve_values: dict = EvalClf(obs=_best_model_results['obs'],
                                              pred=_best_model_results['pred']
                                              ).roc_curve()
            _roc_curve['roc_curve'] = _roc_curve_values['true_positive_rate'][1]
            _roc_curve['baseline'] = _roc_curve_values['false_positive_rate'][1]
            _plot.update({'ROC-AUC Curve:': dict(df=_roc_curve,
                                                 features=['roc_curve', 'baseline'],
                                                 time_features=['baseline'],
                                                 plot_type='line',
                                                 melt=True,
                                                 use_auto_extensions=False,
                                                 # xaxis_label=['False Positive Rate'],
                                                 # yaxis_label=['True Positive Rate'],
                                                 file_path=os.path.join(file_path, 'ga_prediction_roc_auc_curve.html'),
                                                 )
                          })
    return _plot


def evaluate_machine_learning(ml_type: str,
                              target_feature_name: str,
                              prediction_var_name: str,
                              train_data_set_path: str,
                              test_data_set_path: str,
                              metadata_file_path: OutputPath(),
                              metric_file_path: OutputPath(),
                              table_file_path: OutputPath(),
                              output_path_metrics: str,
                              metrics: List[str] = None,
                              labels: List[str] = None,
                              model_id: str = None,
                              val_data_set_path: str = None,
                              sep: str = ',',
                              output_path_confusion_matrix: str = None,
                              output_path_roc_curve: str = None,
                              output_path_metric_table: str = None,
                              output_path_metrics_customized: str = None
                              ) -> NamedTuple('outputs', [('metrics', dict)]):
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

    :param prediction_var_name: str
        Name of the variable containing predictions

    :param train_data_set_path: str
        Complete file path of the training data set

    :param test_data_set_path: str
        Complete file path of the tests data set

    :param metadata_file_path: str
        File path of the metadata output file

    :param metric_file_path: str
        File path of the metric output file

    :param table_file_path: str
        File path of the table output file

    :param output_path_metrics: str
        Path of the output metric results to save

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

    :param output_path_confusion_matrix: str
        Complete file path of the confusion matrix visualization output

    :param output_path_roc_curve: str
        Complete file path of the roc curve visualization output

    :param output_path_metric_table: str
        Complete file path of the metric table visualization output

    :param output_path_metrics_customized: str
        Name of the output S3 bucket

    :return: NamedTuple
        Train and test metric results
    """
    _train_df: pd.DataFrame = pd.read_csv(filepath_or_buffer=train_data_set_path, sep=sep)
    _test_df: pd.DataFrame = pd.read_csv(filepath_or_buffer=test_data_set_path, sep=sep)
    if val_data_set_path is None:
        _val_df: pd.DataFrame = None
    else:
        _val_df: pd.DataFrame = pd.read_csv(filepath_or_buffer=val_data_set_path, sep=sep)
    if metrics is None:
        _metrics: List[str] = ML_METRIC.get(ml_type)
    else:
        _metrics: List[str] = []
        for ml_metric in metrics:
            if ml_metric in ML_METRIC:
                _metrics.append(ml_metric)
            else:
                Log().log(msg=f'Evaluation metric ({ml_metric}) not supported')
        if len(_metrics) == 0:
            raise ModelEvaluationException('No supported evaluation metric found')
    _evaluation: dict = dict(train={}, test={}, val={}, sml_score=None, model_id=model_id, target=target_feature_name)
    for metric in _metrics:
        if ml_type == 'reg':
            _evaluation['train'].update({metric: getattr(EvalReg(obs=_train_df[target_feature_name].values,
                                                                 pred=_train_df[prediction_var_name].values
                                                                 ),
                                                         metric)
                                         })
            _evaluation['test'].update({metric: getattr(EvalReg(obs=_test_df[target_feature_name].values,
                                                                pred=_test_df[prediction_var_name].values
                                                                ),
                                                        metric)
                                        })
            if _val_df is not None:
                _evaluation['val'].update({metric: getattr(EvalReg(obs=_val_df[target_feature_name].values,
                                                                   pred=_val_df[prediction_var_name].values
                                                                   ),
                                                           metric)
                                           })
        else:
            _evaluation['train'].update({metric: getattr(EvalClf(obs=_train_df[target_feature_name].values,
                                                                 pred=_train_df[prediction_var_name].values
                                                                 ),
                                                         metric)
                                         })
            _evaluation['test'].update({metric: getattr(EvalClf(obs=_test_df[target_feature_name].values,
                                                                pred=_test_df[prediction_var_name].values
                                                                ),
                                                        metric)
                                        })
            if metric == 'confusion' and output_path_confusion_matrix is not None:
                if labels is None:
                    _labels: List[str] = [str(label) for label in _test_df[target_feature_name].unique()]
                else:
                    _labels: List[str] = labels
                _df_confusion_matrix: pd.DataFrame = pd.DataFrame(data=EvalClf(obs=_test_df[target_feature_name].values,
                                                                               pred=_test_df[prediction_var_name].values,
                                                                               ).confusion(normalize=None),
                                                                  index=_labels,
                                                                  columns=_labels
                                                                  )
                _df_confusion_matrix.to_csv(path_or_buf=output_path_confusion_matrix, sep=sep, header=False, index=True)
                _confusion_metadata: Dict[str, List[Dict[str, str]]] = _generate_kfp_visualization_template(file_paths=[output_path_confusion_matrix],
                                                                                                            metric_types=['confusion_matrix'],
                                                                                                            target_feature=target_feature_name,
                                                                                                            prediction_feature=prediction_var_name,
                                                                                                            labels=_labels,
                                                                                                            header=None
                                                                                                            )
                file_handler(file_path=metadata_file_path, obj=_confusion_metadata)
            if ml_type == 'clf_binary':
                _metric_value: float = _evaluation['test'][metric] * 100
                if metric == 'accuracy':
                    _metric_type: str = 'accuracy-score'
                elif metric == 'roc_auc':
                    _metric_type: str = 'roc-auc-score'
                    if output_path_roc_curve is not None:
                        _df_roc: pd.DataFrame = pd.DataFrame()
                        _roc_curve: dict = EvalClf(obs=_test_df[target_feature_name].values,
                                                   pred=_test_df[prediction_var_name].values,
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
                        _df_roc.to_csv(path_or_buf=output_path_roc_curve, sep=sep, header=False, index=False)
                        _roc_metadata: Dict[str, List[Dict[str, str]]] = _generate_kfp_visualization_template(file_paths=[output_path_roc_curve],
                                                                                                              metric_types=['roc'],
                                                                                                              target_feature=target_feature_name,
                                                                                                              prediction_feature=prediction_var_name,
                                                                                                              labels=None,
                                                                                                              header=None
                                                                                                              )
                        file_handler(file_path=metadata_file_path, obj=_roc_metadata)
                else:
                    _metric_type: str = None
                if _metric_type is not None:
                    _metric: Dict[str, List[Dict[str, str]]] = _generate_kfp_metric_template(file_paths=[],
                                                                                             metric_types=[_metric_type],
                                                                                             metric_values=[_metric_value],
                                                                                             metric_formats=['PERCENTAGE']
                                                                                             )
                    file_handler(file_path=metric_file_path, obj=_metric)
            if _val_df is not None:
                _evaluation['val'].update({metric: getattr(EvalClf(obs=_val_df[target_feature_name].values,
                                                                   pred=_val_df[prediction_var_name].values
                                                                   ),
                                                           metric)
                                           })
    if SML_SCORE['ml_metric'][ml_type] in list(_evaluation['test'].keys()):
        _evaluation['sml_score'] = sml_fitness_score(ml_metric=tuple([SML_SCORE['ml_metric_best'][ml_type], _evaluation['test'][SML_SCORE['ml_metric'][ml_type]]]),
                                                     train_test_metric=tuple([_evaluation['train'][SML_SCORE['ml_metric'][ml_type]], _evaluation['test'][SML_SCORE['ml_metric'][ml_type]]]),
                                                     train_time_in_seconds=1.0
                                                     )
    file_handler(file_path=output_path_metrics, obj=_evaluation)
    if output_path_metrics_customized is not None:
        save_file_to_s3(file_path=output_path_metrics_customized, obj=_evaluation)
    _skip_metric_summary_table: List[str] = ['classification_report', 'confusion']
    _header: List[str] = ['train', 'test']
    _df_table: pd.DataFrame = pd.DataFrame()
    _train_metric_name: List[str] = []
    _train_metric_value: List[float] = []
    for train_metric_name, train_metric_value in _evaluation['train'].items():
        if train_metric_name not in _skip_metric_summary_table:
            _train_metric_name.append(train_metric_name)
            _train_metric_value.append(train_metric_value)
    _df_table['train'] = _train_metric_value
    _test_metric_value: List[float] = []
    for test_metric_name, test_metric_value in _evaluation['test'].items():
        if test_metric_name not in _skip_metric_summary_table:
            _test_metric_value.append(test_metric_value)
    _df_table['test'] = _test_metric_value
    if val_data_set_path is not None:
        _val_metric_value: List[float] = []
        for val_metric_name, val_metric_value in _evaluation['val'].items():
            if val_metric_name not in _skip_metric_summary_table:
                _val_metric_value.append(val_metric_value)
        _df_table['val'] = _val_metric_value
        _header.append('val')
    _df_table.set_index(keys=_train_metric_name, inplace=True)
    _df_table.to_csv(path_or_buf=output_path_metric_table, sep=sep, header=False, index=True)
    _table_metadata: Dict[str, List[Dict[str, str]]] = _generate_kfp_visualization_template(file_paths=[output_path_metric_table],
                                                                                            metric_types=['table'],
                                                                                            target_feature=target_feature_name,
                                                                                            prediction_feature=prediction_var_name,
                                                                                            labels=labels,
                                                                                            header=_header
                                                                                            )
    file_handler(file_path=table_file_path, obj=_table_metadata)
    return [_evaluation]


if __name__ == '__main__':
    evaluate_machine_learning(ml_type=ARGS.ml_type,
                              target_feature_name=ARGS.target_feature_name,
                              prediction_var_name=ARGS.prediction_var_name,
                              train_data_set_path=ARGS.train_data_set_path,
                              test_data_set_path=ARGS.test_data_set_path,
                              metadata_file_path='metadata_viz.json',
                              metric_file_path='metric_viz.json',
                              table_file_path='table_viz.json',
                              output_path_metrics=ARGS.output_path_metrics,
                              metrics=ARGS.metrics,
                              labels=ARGS.labels,
                              model_id=ARGS.model_id,
                              val_data_set_path=ARGS.val_data_set_path,
                              sep=ARGS.sep,
                              output_path_confusion_matrix=ARGS.output_path_confusion_matrix,
                              output_path_roc_curve=ARGS.output_path_roc_curve,
                              output_path_metric_table=ARGS.output_path_metric_table,
                              output_path_metrics_customized=ARGS.output_path_metrics_customized
                              )
