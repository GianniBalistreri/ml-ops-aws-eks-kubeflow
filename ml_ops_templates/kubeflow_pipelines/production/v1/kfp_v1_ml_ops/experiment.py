"""

Kubeflow Pipeline: Run (recurring) experiment

"""

import boto3
import json
import kfp
import kfp_server_api

from .get_session_cookie_dex import get_istio_auth_session
from kfp.compiler import compiler
from typing import Any, List

AUTH_SERVICE_PROVIDER: List[str] = ['dex']


class KubeflowExperimentException(Exception):
    """
    Class for handling exception for class KubeflowExperiment
    """
    pass


class KubeflowExperiment:
    """
    Class for running Kubeflow experiments
    """
    def __init__(self,
                 kf_url: str,
                 kf_user_name: str,
                 kf_user_pwd: str,
                 kf_user_namespace: str,
                 kf_pipeline_name: str,
                 kf_experiment_name: str,
                 kf_experiment_description: str,
                 kf_experiment_run_name: str,
                 kf_enable_caching: bool,
                 recurring: bool,
                 recurring_start_time: str = None,
                 recurring_end_time: str = None,
                 recurring_interval_second: int = None,
                 recurring_cron_expression: str = '0 0 0 * * *',
                 recurring_max_concurrency: int = 1,
                 recurring_no_catchup: bool = False,
                 recurring_enable: bool = True,
                 recurring_job_name: str = None,
                 recurring_job_description: str = None,
                 auth_service_provider: str = 'dex',
                 service_account: str = None,
                 pipeline_metadata_file_path: str = None
                 ):
        """
        :param kf_url: str
            URL of the Kubeflow deployment

        :param kf_user_name: str
            Kubeflow username

        :param kf_user_pwd: str
            Kubeflow user password

        :param kf_user_namespace: str
            Kubeflow namespace

        :param kf_pipeline_name: str
            Kubeflow pipeline name

        :param kf_experiment_name: str
            Kubeflow experiment name

        :param kf_experiment_description: str
            Kubeflow experiment description

        :param kf_experiment_run_name: str
            Kubeflow experiment run name

        :param kf_enable_caching: bool
            Whether to enable caching for the run or disable

        :param recurring: bool
            Whether to create a recurring run or not

        :param recurring_start_time: str
            Start time of the recurring run

        :param recurring_end_time: str
            End time of the recurring run

        :param recurring_interval_second: int
            Seconds between two recurring runs in for a periodic schedule

        :param recurring_cron_expression: str
            Cron job expression for recurring run

        :param recurring_max_concurrency: int
            Number of jobs that can be run in parallel

        :param recurring_no_catchup: bool
            Whether the recurring run should catch up if behind schedule

        :param recurring_enable: bool
            Whether the recurring run is enabled or disabled

        :param recurring_job_name: str
            Job name of the recurring run

        :param recurring_job_description: str
            Job description of the recurring run

        :param auth_service_provider: str
            Name of the authentication service provider
                -> dex: Dex
                -> cognito: AWS Cognito

        :param service_account: str
            Service account name

        :param pipeline_metadata_file_path: str
            Complete file path of the pipeline metadata
        """
        self.kf_url: str = kf_url
        self.kf_user_name: str = kf_user_name
        self.kf_user_pwd: str = kf_user_pwd
        self.kf_user_namespace: str = kf_user_namespace
        self.kf_pipeline_name: str = kf_pipeline_name
        self.kf_experiment_name: str = kf_experiment_name
        self.kf_experiment_description: str = kf_experiment_description
        self.kf_experiment_run_name: str = kf_experiment_run_name
        self.kf_enable_caching: bool = kf_enable_caching
        self.recurring: bool = recurring
        self.recurring_start_time: str = recurring_start_time
        self.recurring_end_time: str = recurring_end_time
        self.recurring_interval_second: int = recurring_interval_second
        self.recurring_cron_expression: str = recurring_cron_expression
        self.recurring_max_concurrency: int = recurring_max_concurrency
        self.recurring_no_catchup: bool = recurring_no_catchup
        self.recurring_enable: bool = recurring_enable
        self.recurring_job_name: str = recurring_job_name
        self.recurring_job_description: str = recurring_job_description
        self.kfp_client: kfp.Client = None
        if auth_service_provider not in AUTH_SERVICE_PROVIDER:
            raise KubeflowExperimentException(f'Name of the authentication service provider ({auth_service_provider}) not supported')
        self.auth_service_provider: str = auth_service_provider
        self.service_account: str = service_account
        self.pipeline_metadata_file_path: str = pipeline_metadata_file_path

    def _dex_auth(self) -> str:
        """
        Retrieve authentication cookie for dex auth service provider

        :return: str
            Authentication cookie
        """
        return get_istio_auth_session(url=self.kf_url,
                                      username=self.kf_user_name,
                                      password=self.kf_user_pwd
                                      )["session_cookie"]

    def _set_kfp_client(self) -> None:
        """
        Set kfp client
        """
        self.kfp_client = kfp.Client(host=f"{self.kf_url}/pipeline",
                                     client_id=None,
                                     namespace=self.kf_user_namespace,
                                     other_client_id=None,
                                     other_client_secret=None,
                                     existing_token=None,
                                     cookies=self._dex_auth(),
                                     proxy=None,
                                     ssl_ca_cert=None,
                                     kube_context=None,
                                     credentials=None
                                     )

    def main(self, pipeline: Any, arguments: dict = None) -> None:
        """
        Run Kubeflow experiment

        :param pipeline: Any
            Configured pipeline function

        :param arguments: dict
            Pipeline arguments
        """
        self._set_kfp_client()
        compiler.Compiler().compile(pipeline_func=pipeline,
                                    package_path=f"{self.kf_pipeline_name}.yaml",
                                    type_check=True,
                                    pipeline_conf=None
                                    )
        _experiment: kfp_server_api.ApiExperiment = self.kfp_client.create_experiment(name=self.kf_experiment_name,
                                                                                      description=self.kf_experiment_description,
                                                                                      namespace=self.kf_user_namespace
                                                                                      )
        _pipeline_metadata: dict = dict(experiment=dict(created_at=str(_experiment.created_at),
                                                        description=_experiment.description,
                                                        id=_experiment.id,
                                                        name=_experiment.name,
                                                        #resource_references=_experiment.resource_references,
                                                        storage_state=_experiment.storage_state
                                                        )
                                        )
        if self.recurring:
            _pipeline: kfp_server_api.ApiPipeline = self.kfp_client.upload_pipeline(pipeline_package_path=f"{self.kf_pipeline_name}.yaml",
                                                                                    pipeline_name=self.kf_pipeline_name,
                                                                                    description=self.recurring_job_description
                                                                                    )
            _recurring_run: kfp_server_api.ApiJob = self.kfp_client.create_recurring_run(experiment_id=_experiment.id,
                                                                                         job_name=self.recurring_job_name,
                                                                                         description=self.recurring_job_description,
                                                                                         start_time=self.recurring_start_time,
                                                                                         end_time=self.recurring_end_time,
                                                                                         interval_second=self.recurring_interval_second,
                                                                                         cron_expression=self.recurring_cron_expression,
                                                                                         max_concurrency=self.recurring_max_concurrency,
                                                                                         no_catchup=self.recurring_no_catchup,
                                                                                         params=arguments,
                                                                                         pipeline_package_path=f'{self.kf_experiment_name}.yaml',
                                                                                         pipeline_id=_pipeline.id,
                                                                                         version_id=_pipeline.default_version,
                                                                                         enabled=self.recurring_enable,
                                                                                         enable_caching=self.kf_enable_caching,
                                                                                         service_account=self.service_account,
                                                                                         )
            _pipeline_metadata.update({'run': dict(created_at=str(_recurring_run.created_at),
                                                   description=_recurring_run.description,
                                                   id=_recurring_run.id,
                                                   name=_recurring_run.name,
                                                   resource_references=_recurring_run.resource_references,
                                                   status=_recurring_run.status,
                                                   mode=_recurring_run.mode,
                                                   trigger=_recurring_run.trigger,
                                                   updated_at=str(_recurring_run.updated_at)
                                                   ),
                                       'pipeline': dict(name=self.kf_pipeline_name,
                                                        id=_pipeline.id
                                                        )
                                       })
        else:
            _run = self.kfp_client.create_run_from_pipeline_func(pipeline_func=pipeline,
                                                                 arguments={} if arguments is None else arguments,
                                                                 run_name=self.kf_experiment_run_name,
                                                                 experiment_name=self.kf_experiment_name,
                                                                 namespace=self.kf_user_namespace,
                                                                 pipeline_root=None,
                                                                 enable_caching=self.kf_enable_caching,
                                                                 service_account=self.service_account
                                                                 )
            _pipeline_metadata.update({'run': dict(name=self.kf_experiment_run_name,
                                                   id=_run.run_id
                                                   ),
                                       'pipeline': dict(name=self.kf_pipeline_name,
                                                        id=None
                                                        )
                                       })
        _pipeline_metadata.update({'namespace': self.kf_user_namespace})
        if self.pipeline_metadata_file_path is not None:
            _complete_file_path: str = self.pipeline_metadata_file_path.replace('s3://', '')
            _bucket_name: str = _complete_file_path.split('/')[0]
            _file_path: str = _complete_file_path.replace(f'{_bucket_name}/', '')
            _file_type: str = _complete_file_path.split('.')[-1]
            _s3_resource: boto3 = boto3.resource('s3')
            if _file_type == 'json':
                _s3_obj: _s3_resource.Object = _s3_resource.Object(_bucket_name, _file_path)
                _s3_obj.put(Body=json.dumps(obj=_pipeline_metadata))
            else:
                raise KubeflowExperimentException(f'Saving file type ({_file_type}) not supported')
