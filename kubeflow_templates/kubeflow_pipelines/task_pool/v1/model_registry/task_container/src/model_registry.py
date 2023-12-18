"""

Model registry

"""

import pandas as pd

from custom_logger import Log
from typing import List


class ModelRegistryException(Exception):
    """
    Class for handling exception for class ModelRegistry
    """
    pass


class ModelRegistry:
    """
    Class for register trained machine learning models
    """
    def __init__(self, metadata: dict = None):
        """
        :param metadata: dict
            Metadata collection
        """
        self.metadata: dict = metadata

    def _generate_registry_template(self) -> None:
        """
        Generate model registry template
        """
        self.metadata = dict(project_name=[],
                             use_case=[],
                             use_case_type=[],
                             version=[],
                             creation_time=[],
                             artifact_path=[],
                             metric_name=[],
                             train_metric=[],
                             test_metric=[],
                             active=[],
                             replace_time=[],
                             replaced_by=[],
                             ml_type=[],
                             ml_algorithm=[],
                             ml_framework=[],
                             target_feature=[],
                             predictors=[],
                             n_predictors=[],
                             description=[]
                             )

    def _versioning(self) -> None:
        """
        Versioning new registered machine learning model
        """
        _df: pd.DataFrame = pd.DataFrame(data=self.metadata)
        _versions: List[str] = _df.loc[(_df['project_name'] == self.metadata['project_name'][-1]) and (_df['use_case'] == self.metadata['use_case'][-1]) and (_df['use_case_type'] == self.metadata['use_case_type'][-1]), 'version'].values.tolist()
        _new_version: str = f'v{len(_versions) + 1}'
        Log().log(msg=f'Assign model version: {_new_version}')
        self.metadata['version'].append(_new_version)

    def main(self,
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
             description: str
             ) -> None:
        """
        Register machine learning model based on implemented template

        :param project_name:
        :param use_case:
        :param use_case_type:
        :param creation_time:
        :param artifact_path:
        :param metric_name:
        :param train_metric:
        :param test_metric:
        :param ml_type:
        :param ml_algorithm:
        :param ml_framework:
        :param target_feature:
        :param predictors:
        :param description:
        :return:
        """
        self.metadata['project_name'].append(project_name)
        self.metadata['use_case'].append(use_case)
        self.metadata['use_case_type'].append(use_case_type)
        self.metadata['creation_time'].append(creation_time)
        self.metadata['artifact_path'].append(artifact_path)
        self.metadata['metric_name'].append(metric_name)
        self.metadata['train_metric'].append(train_metric)
        self.metadata['test_metric'].append(test_metric)
        self.metadata['ml_type'].append(ml_type)
        self.metadata['ml_algorithm'].append(ml_algorithm)
        self.metadata['ml_framework'].append(ml_framework)
        self.metadata['target_feature'].append(target_feature)
        self.metadata['predictors'].append(','.join(predictors))
        self.metadata['predictors'].append(len(predictors))
        self.metadata['description'].append(description)
        self._versioning()
