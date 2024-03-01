"""

Task: ... (Function to run in container)

"""

import argparse
import boto3
import json
import urllib3

from custom_logger import Log
from kubeflow_exit_handler import KubeflowExitHandler
from typing import NamedTuple

PARSER = argparse.ArgumentParser(description="slack messaging and alerting")
PARSER.add_argument('-exit_handler', type=int, required=True, default=None, help='whether task should act like an exit handler or not')
PARSER.add_argument('-aws_region', type=str, required=True, default=None, help='AWS region code')
PARSER.add_argument('-secret_name', type=str, required=True, default=None, help='secret name of the secret manager entry containing the destined slack channel web token')
PARSER.add_argument('-header', type=str, required=False, default=None, help='pre-defined header text')
PARSER.add_argument('-msg', type=str, required=False, default=None, help='pre-defined message')
PARSER.add_argument('-status', type=int, required=False, default=None, help='pipeline status code')
PARSER.add_argument('-pipeline_metadata_file_path', type=str, required=False, default=None, help='complete file path of the pipeline metadata')
ARGS = PARSER.parse_args()


class SlackAlertingException(Exception):
    """
    Class for handling exceptions for function slack_alerting
    """
    pass


def slack_alerting(exit_handler: bool,
                   aws_region: str,
                   secret_name: str,
                   header: str = None,
                   msg: str = None,
                   status: int = None,
                   pipeline_metadata_file_path: str = None
                   ) -> NamedTuple('outputs', [('header', str),
                                               ('message', str),
                                               ('response_status_code', int)
                                               ]
                                   ):
    """
    Send messages and alerts to dedicated Slack channel

    :param exit_handler: bool
        Whether task should act like an exit handler

    :param aws_region: str
        Code of the AWS region

    :param secret_name: str
        Secret name of the secret manager entry containing Slack channel

    :param header: str
        Pre-defined header text

    :param msg: str
        Pre-defined message

    :param status: int
        Status code of the message:
            -> 0: error
            -> 1: warning
            -> 2: info
            -> 3: succeed

    :param pipeline_metadata_file_path: str
            Complete file path of the pipeline metadata

    :return: NamedTuple
        Status code of the request, header and message text
    """
    _client: boto3 = boto3.client('secretsmanager', region_name=aws_region)
    _secret_value: dict = _client.get_secret_value(SecretId=secret_name)
    _slack_channel_name: str = _secret_value['SecretString'].split('"')[1]
    _slack_url: str = _secret_value['SecretString'].split('"')[3]
    _http: urllib3.PoolManager = urllib3.PoolManager()
    if exit_handler:
        _kubeflow_exit_handler: KubeflowExitHandler = KubeflowExitHandler(pipeline_metadata_file_path=pipeline_metadata_file_path)
        _pipeline_status: dict = _kubeflow_exit_handler.main()
        _status: int = _pipeline_status.get('status')
        _header: str = _pipeline_status.get('header')
        _message: str = _pipeline_status.get('message')
    else:
        _status: int = status
        _header: str = header
        _message: str = msg
    if _status == 0:
        _status_symbol: str = ':x:'
    elif _status == 1:
        _status_symbol: str = ':warning: Alarm:'
    elif _status == 2:
        _status_symbol: str = ':information_source:'
    elif _status == 3:
        _status_symbol: str = ':large_green_circle:'
    else:
        raise SlackAlertingException(f'Pipeline status code ({_status}) not supported')
    _slack_msg: dict = {
        "type": "home",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain.text",
                    "text": f"{_status_symbol} {_header}"
                }
            },
            {
                "type": "divider"
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"_{_message}_"
                },
                "block_id": "text1"
            },
            {
                "type": "divider"
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"Region: *{aws_region}*"
                    }
                ]
            }
        ]
    }
    _encoded_msg: bytes = json.dumps(_slack_msg).encode('utf-8')
    _response = _http.request(method='POST', url=_slack_url, body=_encoded_msg)
    Log().log(msg=f'Slack message send to channel "{_slack_channel_name}" with response status "{_response.status}"')
    return [_header, _message, _response.status]


if __name__ == '__main__':
    slack_alerting(exit_handler=bool(ARGS.exit_handler),
                   aws_region=ARGS.aws_region,
                   secret_name=ARGS.secret_name,
                   header=ARGS.header,
                   msg=ARGS.msg,
                   status=ARGS.status,
                   pipeline_metadata_file_path=ARGS.pipeline_metadata_file_path
                   )
