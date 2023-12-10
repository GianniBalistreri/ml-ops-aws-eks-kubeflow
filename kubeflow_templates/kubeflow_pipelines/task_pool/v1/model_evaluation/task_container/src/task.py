"""

Task: ... (Function to run in container)

"""

import argparse
import boto3
import json
import pandas as pd

from evaluate_machine_learning import EvalClf, EvalReg, ML_METRIC, sml_fitness_score, SML_SCORE
from kfp.v2.dsl import ClassificationMetrics, Metrics, Output
from typing import NamedTuple, List


PARSER = argparse.ArgumentParser(description="calculate metrics for evaluating machine learning models")
PARSER.add_argument('-ml_type', type=str, required=True, default=None, help='machine learning type')
PARSER.add_argument('-target_feature_name', type=str, required=True, default=None, help='name of the target feature')
PARSER.add_argument('-prediction_feature_name', type=str, required=True, default=None, help='name of the prediction variable')
PARSER.add_argument('-output_path_metrics', type=str, required=True, default=None, help='output file path of the metric')
PARSER.add_argument('-train_data_set_path', type=str, required=True, default=None, help='complete file path of the training data set')
PARSER.add_argument('-test_data_set_path', type=str, required=True, default=None, help='complete file path of the test data set')
PARSER.add_argument('-val_data_set_path', type=str, required=False, default=None, help='complete file path of the validation data set')
PARSER.add_argument('-prediction_feature_name', type=str, required=True, default=None, help='name of the prediction variable')
PARSER.add_argument('-sep', type=str, required=False, default=',', help='column separator')
PARSER.add_argument('-metrics', type=str, required=False, default=None, help='metrics to calculate')
PARSER.add_argument('-output_bucket_name', type=str, required=False, default=None, help='output S3 bucket name')
ARGS = PARSER.parse_args()


def evaluate_machine_learning(ml_type: str,
                              target_feature_name: str,
                              prediction_var_name: str,
                              output_path_metrics: str,
                              train_data_set_path: str,
                              test_data_set_path: str,
                              val_data_set_path: str = None,
                              sep: str = ',',
                              metrics: List[str] = None,
                              output_bucket_name: str = None
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

    :param output_path_metrics: str
        Path of the output metric results to save

    :param train_data_set_path: str
        Complete file path of the training data set

    :param test_data_set_path: str
        Complete file path of the tests data set

    :param val_data_set_path: str
        Complete file path of the validation data set

    :param sep: str
        Separator

    :param metrics: List[str]
        Abbreviated names of metrics to apply

    :param output_bucket_name: str
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
        _metrics: List[str] = metrics
    _evaluation: dict = dict(train={}, test={}, val={})
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
    with open(output_path_metrics.split('/')[-1], 'w') as _file:
        json.dump(_evaluation, _file)
    if output_bucket_name is not None:
        _s3_resource: boto3 = boto3.resource('s3')
        _s3_obj: _s3_resource.Object = _s3_resource.Object(output_bucket_name, output_path_metrics)
        _s3_obj.put(Body=(bytes(json.dumps(_evaluation).encode('UTF-8'))))
    return [_evaluation]


if __name__ == '__main__':
    evaluate_machine_learning(ml_type=ARGS.ml_type,
                              target_feature_name=ARGS.target_feature_name,
                              prediction_var_name=ARGS.prediction_var_name,
                              output_path_metrics=ARGS.output_path_metrics,
                              train_data_set_path=ARGS.train_data_set_path,
                              test_data_set_path=ARGS.test_data_set_path,
                              val_data_set_path=ARGS.val_data_set_path,
                              sep=ARGS.sep,
                              metrics=ARGS.metrics,
                              output_bucket_name=ARGS.output_bucket_name
                              )
