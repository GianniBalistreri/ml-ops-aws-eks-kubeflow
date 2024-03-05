"""

Task: ... (Function to run in container)

"""

import argparse
import boto3
import json
import urllib3

from custom_logger import Log
from file_handler import file_handler
from kubeflow_exit_handler import KubeflowExitHandler
from typing import NamedTuple

PARSER = argparse.ArgumentParser(description="slack messaging and alerting")
PARSER.add_argument('-exit_handler', type=int, required=True, default=None, help='whether task should act like an exit handler or not')
PARSER.add_argument('-aws_region', type=str, required=True, default=None, help='AWS region code')
PARSER.add_argument('-secret_name_slack', type=str, required=True, default=None, help='secret name of the secret manager entry containing the destined slack channel web token')
PARSER.add_argument('-header', type=str, required=False, default=None, help='pre-defined header text')
PARSER.add_argument('-msg', type=str, required=False, default=None, help='pre-defined message')
PARSER.add_argument('-footer', type=str, required=False, default=None, help='pre-defined footer text')
PARSER.add_argument('-status', type=int, required=False, default=None, help='pipeline status code')
PARSER.add_argument('-pipeline_metadata_file_path', type=str, required=False, default=None, help='complete file path of the pipeline metadata')
PARSER.add_argument('-kf_url', type=str, required=False, default=None, help='kubeflow url')
PARSER.add_argument('-kf_user_namespace', type=str, required=False, default=None, help='kubeflow user namespace')
PARSER.add_argument('-secret_name_kf', type=str, required=False, default=None, help='secret name of the secret manager entry containing the kubeflow user credentials')
PARSER.add_argument('-output_file_path_header', type=str, required=False, default=None, help='file path of the header output')
PARSER.add_argument('-output_file_path_message', type=str, required=False, default=None, help='file path of the message output')
PARSER.add_argument('-output_file_path_footer', type=str, required=False, default=None, help='file path of the footer output')
PARSER.add_argument('-output_file_path_response_status_code', type=str, required=False, default=None, help='file path of the response status code output')
ARGS = PARSER.parse_args()


class SlackAlertingException(Exception):
    """
    Class for handling exceptions for function slack_alerting
    """
    pass


def slack_alerting(exit_handler: bool,
                   aws_region: str,
                   secret_name_slack: str,
                   output_file_path_header: str,
                   output_file_path_message: str,
                   output_file_path_footer: str,
                   output_file_path_response_status_code: str,
                   header: str = None,
                   msg: str = None,
                   footer: str = None,
                   status: int = None,
                   pipeline_metadata_file_path: str = None,
                   kf_url: str = None,
                   kf_user_namespace: str = None,
                   secret_name_kf: str = None
                   ) -> NamedTuple('outputs', [('header', str),
                                               ('message', str),
                                               ('footer', str),
                                               ('response_status_code', int)
                                               ]
                                   ):
    """
    Send messages and alerts to dedicated Slack channel

    :param exit_handler: bool
        Whether task should act like an exit handler

    :param aws_region: str
        Code of the AWS region

    :param secret_name_slack: str
        Secret name of the secret manager entry containing Slack channel credentials

    :param output_file_path_header: str
        File path of the header output

    :param output_file_path_message: str
        File path of the message output

    :param output_file_path_footer: str
        File path of the footer output

    :param output_file_path_response_status_code: str
        File path of the response status code output

    :param header: str
        Pre-defined header text

    :param msg: str
        Pre-defined message

    :param footer: str
        Pre-defined footer text

    :param status: int
        Status code of the message:
            -> 0: error
            -> 1: warning
            -> 2: info
            -> 3: succeed

    :param pipeline_metadata_file_path: str
            Complete file path of the pipeline metadata

    :param kf_url: str
            URL of the Kubeflow deployment

    :param kf_user_namespace: str
        Kubeflow namespace

    :param secret_name_kf: str
        Secret name of the secret manager entry containing Kubeflow user credentials

    :return: NamedTuple
        Status code of the request, header, footer and message text
    """
    _client: boto3 = boto3.client('secretsmanager', region_name=aws_region)
    _secret_value: dict = _client.get_secret_value(SecretId=secret_name_slack)
    _slack_channel_name: str = _secret_value['SecretString'].split('"')[1]
    _slack_url: str = _secret_value['SecretString'].split('"')[3]
    _http: urllib3.PoolManager = urllib3.PoolManager()
    if exit_handler:
        _kubeflow_exit_handler: KubeflowExitHandler = KubeflowExitHandler(pipeline_metadata_file_path=pipeline_metadata_file_path,
                                                                          kf_url=kf_url,
                                                                          kf_user_namespace=kf_user_namespace,
                                                                          secret_name=secret_name_kf,
                                                                          aws_region=aws_region
                                                                          )
        _pipeline_status: dict = _kubeflow_exit_handler.main()
        _status: int = _pipeline_status.get('status')
        _header: str = _pipeline_status.get('header')
        _message: str = _pipeline_status.get('message')
        _footer: str = _pipeline_status.get('footer')
    else:
        _status: int = status
        _header: str = header
        _message: str = msg
        _footer: str = footer
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
                    "type": "plain_text",
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
                    "text": f"{_message}"
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
                        "text": f"*{_footer}*"
                    }
                ]
            }
        ]
    }
    _encoded_msg: bytes = json.dumps(_slack_msg).encode('utf-8')
    _response = _http.request(method='POST', url=_slack_url, body=_encoded_msg)
    Log().log(msg=f'Slack message send to channel "{_slack_channel_name}" with response status "{_response.status}"')
    file_handler(file_path=output_file_path_header, obj=_header)
    file_handler(file_path=output_file_path_message, obj=_message)
    file_handler(file_path=output_file_path_footer, obj=_footer)
    file_handler(file_path=output_file_path_response_status_code, obj=_response.status)
    return [_header, _message, _footer, _response.status]


if __name__ == '__main__':
    slack_alerting(exit_handler=bool(ARGS.exit_handler),
                   aws_region=ARGS.aws_region,
                   secret_name_slack=ARGS.secret_name_slack,
                   output_file_path_header=ARGS.output_file_path_header,
                   output_file_path_message=ARGS.output_file_path_message,
                   output_file_path_footer=ARGS.output_file_path_footer,
                   output_file_path_response_status_code=ARGS.output_file_path_response_status_code,
                   header=ARGS.header,
                   msg=ARGS.msg,
                   footer=ARGS.footer,
                   status=ARGS.status,
                   pipeline_metadata_file_path=ARGS.pipeline_metadata_file_path,
                   kf_url=ARGS.kf_url,
                   kf_user_namespace=ARGS.kf_user_namespace,
                   secret_name_kf=ARGS.secret_name_kf,
                   )
