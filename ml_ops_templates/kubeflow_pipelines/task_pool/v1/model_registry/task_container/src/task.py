"""

Task: ... (Function to run in container)

"""

import argparse
import ast

from aws import file_exists, load_file_from_s3, save_file_to_s3
from model_registry import ModelRegistry
from file_handler import file_handler
from typing import List, NamedTuple


PARSER = argparse.ArgumentParser(description="model registry")
PARSER.add_argument('-registry_file_path', type=str, required=True, default=None, help='file path of the model registry')
PARSER.add_argument('-project_name', type=str, required=True, default=None, help='project name')
PARSER.add_argument('-use_case', type=str, required=True, default=None, help='name of project use case')
PARSER.add_argument('-use_case_type', type=str, required=True, default=None, help='name of the project use case type')
PARSER.add_argument('-creation_time', type=str, required=True, default=None, help='creation time stamp of the model artifact')
PARSER.add_argument('-artifact_path', type=str, required=True, default=None, help='file path of the stored model artifact')
PARSER.add_argument('-metric_name', type=str, required=False, default=None, help='name of the main metric used for model evaluation')
PARSER.add_argument('-train_metric', type=str, required=False, default=None, help='metric value of the evaluation of the training data')
PARSER.add_argument('-test_metric', type=str, required=False, default=None, help='metric value of the evaluation of the testing data')
PARSER.add_argument('-ml_type', type=str, required=True, default=None, help='name of the used machine learning type')
PARSER.add_argument('-ml_algorithm', type=str, required=True, default=None, help='name of the used machine learning algorithm')
PARSER.add_argument('-ml_framework', type=str, required=True, default=None, help='name of the used machine learning framework')
PARSER.add_argument('-target_feature', type=str, required=False, default=None, help='name of the target feature')
PARSER.add_argument('-predictors', nargs='+', required=False, default=None, help='name of the predictors')
PARSER.add_argument('-description', type=str, required=False, default=None, help='detailed model description')
PARSER.add_argument('-output_file_path_version', type=str, required=True, default=None, help='file path of defined model version output')
PARSER.add_argument('-output_file_path_registry', type=str, required=True, default=None, help='file path of the model registry output')
PARSER.add_argument('-output_file_path_registry_customized', type=str, required=False, default=None, help='complete customized file path of the model registry output')
ARGS = PARSER.parse_args()


def model_registry(registry_file_path: str,
                   project_name: str,
                   use_case: str,
                   use_case_type: str,
                   creation_time: str,
                   artifact_path: str,
                   metric_name: str,
                   train_metric: str,
                   test_metric: str,
                   ml_type: str,
                   ml_algorithm: str,
                   ml_framework: str,
                   target_feature: str,
                   predictors: List[str],
                   description: str,
                   output_file_path_version: str,
                   output_file_path_registry: str,
                   output_file_path_registry_customized: str = None,
                   ) -> NamedTuple('outputs', [('registry', dict),
                                               ('version', str)
                                               ]
                                   ):
    """
    Register new machine learning model

    :param registry_file_path: str
        Complete file path of the model registry

    :param project_name: str
        Name of the project

    :param use_case: str
        Name of the use case

    :param use_case_type: str
        Name of the use case type
            -> batch: Batch prediction scenario
            -> real_time: Real-time prediction scenario

    :param creation_time: str
        Creation timestamp

    :param artifact_path: str
        Complete file path of the stored model artifact

    :param metric_name: str
        Name of the main metric

    :param train_metric: str
        Metric value of the training data evaluation

    :param test_metric: str
        Metric value of the testing data evaluation

    :param ml_type: str
        Abbreviated name of the machine learning type
            -> reg: Regression
            -> clf_binary: Binary classification
            -> clf_multi: Multi-Classification
            -> cluster: Clustering

    :param ml_algorithm: str
        Abbreviated name of the machine learning algorithm

    :param ml_framework: str
        Name of the machine learning framework

    :param target_feature: str
        Name of the target feature

    :param predictors: List[str]
        Name of the predictors

    :param description: str
        Detailed description

    :param output_file_path_version: str


    :param output_file_path_registry: str
        -

    :param output_file_path_registry_customized: str

    :return: NamedTuple

    """
    if file_exists(file_path=registry_file_path):
        _metadata: dict = load_file_from_s3(file_path=registry_file_path)
    else:
        _metadata: dict = None
    _model_registry: ModelRegistry = ModelRegistry(metadata=_metadata)
    _model_registry.main(project_name=project_name,
                         use_case=use_case,
                         use_case_type=use_case_type,
                         creation_time=creation_time,
                         artifact_path=artifact_path,
                         metric_name=metric_name,
                         train_metric=train_metric,
                         test_metric=test_metric,
                         ml_type=ml_type,
                         ml_algorithm=ml_algorithm,
                         ml_framework=ml_framework,
                         target_feature=target_feature,
                         predictors=predictors,
                         description=description
                         )
    for file_path, obj in [(output_file_path_registry, _model_registry.metadata),
                           (output_file_path_version, _model_registry.metadata['version'][-1])
                           ]:
        file_handler(file_path=file_path, obj=obj)
    if output_file_path_registry_customized is not None:
        save_file_to_s3(file_path=output_file_path_registry_customized, obj=_model_registry.metadata)
    return [_model_registry.metadata,
            _model_registry.metadata['version'][-1]
            ]


if __name__ == '__main__':
    if ARGS.predictors:
        ARGS.predictors = ast.literal_eval(ARGS.predictors[0])
    model_registry(registry_file_path=ARGS.registry_file_path,
                   project_name=ARGS.project_name,
                   use_case=ARGS.use_case,
                   use_case_type=ARGS.use_case_type,
                   creation_time=ARGS.creation_time,
                   artifact_path=ARGS.artifact_path,
                   metric_name=ARGS.metric_name,
                   train_metric=ARGS.train_metric,
                   test_metric=ARGS.test_metric,
                   ml_type=ARGS.ml_type,
                   ml_algorithm=ARGS.ml_algorithm,
                   ml_framework=ARGS.ml_framework,
                   target_feature=ARGS.target_feature,
                   predictors=ARGS.predictors,
                   description=ARGS.description,
                   output_file_path_version=ARGS.output_file_path_version,
                   output_file_path_registry=ARGS.output_file_path_registry,
                   output_file_path_registry_customized=ARGS.output_file_path_registry_customized
                   )
