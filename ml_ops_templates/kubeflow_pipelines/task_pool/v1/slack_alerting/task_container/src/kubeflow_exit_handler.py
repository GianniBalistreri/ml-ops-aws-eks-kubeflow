"""

Exit handler for Kubeflow pipelines

"""

import boto3

from aws import file_exists, load_file_from_s3
from custom_logger import Log
from datetime import datetime
from get_session_cookie_dex import get_istio_auth_session
from kfp import Client
from typing import List


class KubeflowExitHandler:
    """
    Class for Kubeflow pipeline exit handling
    """
    def __init__(self,
                 pipeline_metadata_file_path: str,
                 kf_url: str,
                 kf_user_namespace: str,
                 secret_name: str,
                 aws_region: str
                 ):
        """
        :param pipeline_metadata_file_path: str
            Complete file path of the pipeline metadata

        :param kf_url: str
            URL of the Kubeflow deployment

        :param kf_user_namespace: str
            Kubeflow namespace

        :param secret_name: str
            Secret name of the secret manager entry containing Slack channel

        :param aws_region: str
            Code of the AWS region
        """
        _client: boto3 = boto3.client('secretsmanager', region_name=aws_region)
        _secret_value: dict = _client.get_secret_value(SecretId=secret_name)
        _kf_user_name: str = _secret_value['SecretString'].split('"')[1]
        _kf_pwd: str = _secret_value['SecretString'].split('"')[3]
        self.kfp_client: Client = Client(host=f'{kf_url}/pipeline',
                                         client_id=None,
                                         namespace=kf_user_namespace,
                                         other_client_id=None,
                                         other_client_secret=None,
                                         existing_token=None,
                                         cookies=get_istio_auth_session(url=kf_url,
                                                                        username=_kf_user_name,
                                                                        password=_kf_pwd
                                                                        )["session_cookie"],
                                         proxy=None,
                                         ssl_ca_cert=None,
                                         kube_context=None,
                                         credentials=None,
                                         ui_host=None
                                         )
        self.pipeline_metadata_file_path: str = pipeline_metadata_file_path

    def main(self) -> dict:
        """
        Get status of the current pipeline

        :return: dict
            Pipeline status information
        """
        _pipeline_status: dict = dict(status=2,
                                      header='Kubeflow pipeline finished',
                                      message='-',
                                      footer='-'
                                      )
        if self.pipeline_metadata_file_path is not None:
            if file_exists(file_path=self.pipeline_metadata_file_path):
                _pipeline_metadata: dict = load_file_from_s3(file_path=self.pipeline_metadata_file_path)
                Log().log(msg=f'Load pipeline metadata: {self.pipeline_metadata_file_path}')
                Log().log(msg=f'Experiment information: {_pipeline_metadata.get("experiment")}')
                Log().log(msg=f'Run id: {_pipeline_metadata["run"]["id"]}')
                _end_time: str = str(datetime.now()).split('.')[0]
                Log().log(msg=f'Finished pipeline tasks: {_end_time}')
                _pipeline_status['footer'] = f'Namespace: {_pipeline_metadata.get("namespace")}'
                _pipeline_status['status'] = 3
                _succeeds: int = 0
                _failures: int = 0
                _omits: int = 0
                _pending: int = 0
                _running: int = 0
                _failed_task_name: List[str] = []
                _failed_task_display_name: List[str] = []
                _error_message: List[str] = []
                _run_children = self.kfp_client.get_run(run_id=_pipeline_metadata['run']['id']).pipeline_runtime.workflow_manifest.split('children')
                for c in range(0, len(_run_children), 1):
                    if c == 0:
                        continue
                    _phases = _run_children[c].split('phase')
                    for p in range(0, len(_phases), 1):
                        if _phases[p].find('displayName') >= 0:
                            if p + 1 <= len(_phases):
                                _task_name: str = _phases[p].split('name":')[1].split(',')[0].replace('"', '')
                                Log().log(msg=f'Task name: {_task_name}')
                                _task_display_name: str = _phases[p].split('displayName":')[1].split(',')[0].replace('"', '')
                                Log().log(msg=f'Task display name: {_task_display_name}')
                                _phase: str = _phases[p + 1].split('"')[2]
                                Log().log(msg=f'Phase: {_phase}')
                                if _phase == 'Succeeded':
                                    _succeeds += 1
                                elif _phase == 'Failed':
                                    _pipeline_status['status'] = 0
                                    if _phases[p + 1].find('message":') >= 0:
                                        _failures += 1
                                        _failed_task_name.append(_task_name)
                                        _failed_task_display_name.append(_task_display_name)
                                        _error_message.append(_phases[p + 1].split('message":')[1].split(',')[0].replace('"', ''))
                                elif _phase == 'Omitted':
                                    _omits += 1
                                elif _phase == 'Pending':
                                    _pending += 1
                                elif _phase == 'Running':
                                    _running += 1
                if _pipeline_status['status'] == 0:
                    _pipeline_status['header'] = 'Kubeflow pipeline aborted'
                    _failure_info: str = 'Failed tasks:'
                    for failure in range(0, _failures, 1):
                        _failure_info = f'{_failure_info}\n{failure + 1})\n' \
                                        f'Task Name: {_failed_task_name[failure]}\n' \
                                        f'Task Display Name: {_failed_task_display_name[failure]}\n' \
                                        f'Error Message: {_error_message[failure]}\n\n'
                else:
                    _pipeline_status['header'] = 'Kubeflow pipeline succeeded'
                    _failure_info: str = ''
                _components: int = _succeeds + _failures + _omits + _pending + _running
                _message: str = f'Experiment:\n' \
                                f'Name: {_pipeline_metadata["experiment"].get("name")}\n' \
                                f'Description: {_pipeline_metadata["experiment"].get("description")}\n' \
                                f'ID: {_pipeline_metadata["experiment"].get("id")}\n' \
                                f'Created at: {_pipeline_metadata["experiment"].get("created_at").split("+")[0]}\n' \
                                f'\n' \
                                f'Run:\n' \
                                f'Name: {_pipeline_metadata["run"].get("name")}\n' \
                                f'ID: {_pipeline_metadata["run"].get("id")}\n' \
                                f'Finish: {_end_time}\n' \
                                f'\n' \
                                f'Pipeline:\n' \
                                f'Name: {_pipeline_metadata["pipeline"].get("name")}\n' \
                                f'ID: {_pipeline_metadata["pipeline"].get("id")}\n' \
                                f'Components: {_components}\n' \
                                f'\n' \
                                f'Task succeeded: {_succeeds}\n' \
                                f'Task failed: {_failures}\n' \
                                f'Task omitted: {_omits}\n' \
                                f'Task pending: {_pending}\n' \
                                f'Task running: {_running}\n' \
                                f'\n{_failure_info}'
                _pipeline_status.update({'message': _message})
        return _pipeline_status
