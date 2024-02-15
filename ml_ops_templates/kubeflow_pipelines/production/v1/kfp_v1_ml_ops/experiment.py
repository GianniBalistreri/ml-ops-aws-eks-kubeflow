"""

Kubeflow Pipeline: Run (recurring) experiment

"""


import kfp
from kfp.compiler import compiler
from shopware_kfp_utils.get_session_cookie_dex import get_istio_auth_session
from typing import List

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
                 recurring_no_catchup: bool = False,
                 recurring_enable: bool = True,
                 recurring_job_name: str = None,
                 recurring_job_description: str = None,
                 auth_service_provider: str = 'dex'
                 ):
        """
        :param kf_url:
        :param kf_user_name:
        :param kf_user_pwd:
        :param kf_user_namespace:
        :param kf_pipeline_name:
        :param kf_experiment_name:
        :param kf_experiment_description:
        :param kf_experiment_run_name:
        :param kf_enable_caching:
        :param recurring:
        :param recurring_start_time:
        :param recurring_end_time:
        :param recurring_interval_second:
        :param recurring_cron_expression:
        :param recurring_no_catchup:
        :param recurring_enable: bool
        :param recurring_job_name: str
        :param recurring_job_description: str
        :param auth_service_provider: str
            Name of the authentication service provider
                -> dex: Dex
                -> cognito: AWS Cognito
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
        self.recurring_no_catchup: bool = recurring_no_catchup
        self.recurring_enable: bool = recurring_enable
        self.recurring_job_name: str = recurring_job_name
        self.recurring_job_description: str = recurring_job_description
        self.kfp_client: kfp.Client = None
        if auth_service_provider not in AUTH_SERVICE_PROVIDER:
            raise KubeflowExperimentException(f'Name of the authentication service provider ({auth_service_provider}) not supported')
        self.auth_service_provider: str = auth_service_provider

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

    def main(self, pipeline) -> None:
        """
        Run Kubeflow experiment

        :param pipeline:
            Configured pipeline function
        """
        self._set_kfp_client()
        compiler.Compiler().compile(pipeline_func=pipeline,
                                    package_path=f"{self.kf_pipeline_name}.yaml",
                                    type_check=True,
                                    pipeline_conf=None
                                    )
        self.kfp_client.create_experiment(name=self.kf_experiment_name,
                                          description=self.kf_experiment_description,
                                          namespace=self.kf_user_namespace
                                          )
        if self.recurring:
            _experiment_id: str = self.kfp_client.get_experiment(experiment_id=None,
                                                                 experiment_name=self.kf_experiment_name,
                                                                 namespace=self.kf_user_namespace
                                                                 ).id
            self.kfp_client.create_recurring_run(experiment_id=_experiment_id,
                                                 job_name=self.recurring_job_name,
                                                 description=self.recurring_job_description,
                                                 start_time=self.recurring_start_time,
                                                 end_time=self.recurring_end_time,
                                                 interval_second=self.recurring_interval_second,
                                                 cron_expression=self.recurring_cron_expression,
                                                 max_concurrency=1,
                                                 no_catchup=self.recurring_no_catchup,
                                                 params=None,
                                                 pipeline_package_path=f'{self.kf_experiment_name}.yaml',
                                                 pipeline_id=None,
                                                 version_id=None,
                                                 enabled=self.recurring_enable,
                                                 enable_caching=self.kf_enable_caching,
                                                 service_account=None
                                                 )
        else:
            self.kfp_client.create_run_from_pipeline_func(pipeline_func=pipeline,
                                                          arguments={},
                                                          run_name=self.kf_experiment_run_name,
                                                          experiment_name=self.kf_experiment_name,
                                                          namespace=self.kf_user_namespace,
                                                          pipeline_root=None,
                                                          enable_caching=self.kf_enable_caching,
                                                          service_account=None
                                                          )
