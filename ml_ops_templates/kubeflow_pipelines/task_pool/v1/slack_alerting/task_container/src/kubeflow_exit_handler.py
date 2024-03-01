"""

Exit handler for Kubeflow pipelines

"""

import os

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
    def __init__(self, pipeline_metadata_file_path: str):
        """
        :param pipeline_metadata_file_path: str
            Complete file path of the pipeline metadata
        """
        self.kfp_client: Client = Client(host=os.getenv('KF_URL'),
                                         client_id=None,
                                         namespace=os.getenv('KF_NAMESPACE'),
                                         other_client_id=None,
                                         other_client_secret=None,
                                         existing_token=None,
                                         cookies=get_istio_auth_session(url=os.getenv('KF_URL'),
                                                                        username=os.getenv('KF_USER_NAME'),
                                                                        password=os.getenv('KF_USER_PWD')
                                                                        ),
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
        _pipeline_status: dict = dict(status=2, header='Kubeflow pipeline finished', message='')
        if self.pipeline_metadata_file_path is not None:
            if file_exists(file_path=self.pipeline_metadata_file_path):
                _pipeline_metadata: dict = load_file_from_s3(file_path=self.pipeline_metadata_file_path)
                Log().log(msg=f'Load pipeline metadata: {self.pipeline_metadata_file_path}')
                Log().log(msg=f'Experiment information: {_pipeline_metadata.get("experiment")}')
                Log().log(msg=f'Run id: {_pipeline_metadata.get("run")}')
                _end_time: str = str(datetime.now()).split('.')[0]
                Log().log(msg=f'Finished pipeline tasks: {_end_time}')
                _pipeline_status['status'] = 1
                _succeeds: int = 0
                _failures: int = 0
                _pending: int = 0
                _running: int = 0
                _failed_task_name: List[str] = []
                _failed_task_display_name: List[str] = []
                _error_message: List[str] = []
                for i, child in enumerate(self.kfp_client.get_run(run_id=_pipeline_metadata.get('run')).pipeline_runtime.workflow_manifest.split('children')):
                    if i == 0:
                        continue
                    _task_name: str = child.split('name":')[1].split(',')[0].replace('"', '')
                    Log().log(msg=f'Task name: {_task_name}')
                    _task_display_name: str = child.split('displayName":')[1].split(',')[0].replace('"', '')
                    Log().log(msg=f'Task display name: {_task_display_name}')
                    _phase: str = child.split('phase')[-1].split('","')[0].replace('":"', '')
                    Log().log(msg=f'Phase: {_phase}')
                    if _phase == 'Succeeded':
                        _succeeds += 1
                    elif _phase == 'Failed':
                        _failures += 1
                        _pipeline_status['status'] = 0
                        _failed_task_name.append(_task_name)
                        _failed_task_display_name.append(_task_display_name)
                        _error_message.append(child.split('message":')[1].split(',')[0].replace('"', ''))
                        Log().log(msg=f'Error message: {_error_message[-1]}')
                    elif _phase == 'Pending':
                        _pending += 1
                    elif _phase == 'Running':
                        _running += 1
                if _pipeline_status['status'] == 0:
                    _pipeline_status['header'] = 'Kubeflow pipeline aborted'
                    _failure_info: str = 'Failed tasks:\n'
                    for failure in range(0, _failures, 1):
                        _failure_info = f'{failure + 1})\n{_failure_info}Task Name: {_failed_task_name[failure]}\nTask Display Name: {_failed_task_display_name[failure]}\nError Message: {_error_message[failure]}\n\n'
                else:
                    _pipeline_status['header'] = 'Kubeflow pipeline succeeded'
                    _failure_info: str = ''
                _message: str = f'Experiment: {_pipeline_metadata["experiment"].get("name")}\n' \
                                f'Run: {_pipeline_metadata.get("run")}\n' \
                                f'Start: {_pipeline_metadata["experiment"].get("created_at").split("+")[0]}\n' \
                                f'Finish: {_end_time}\n\n' \
                                f'Task succeeded: {_succeeds}\n' \
                                f'Task failed: {_failures}\n' \
                                f'Task pending: {_pending}\n' \
                                f'Task running: {_running}\n{_failure_info}'
                _pipeline_status.update({'message': _message})
        return _pipeline_status
