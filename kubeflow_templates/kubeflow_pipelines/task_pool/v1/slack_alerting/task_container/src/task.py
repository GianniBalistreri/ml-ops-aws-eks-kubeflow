"""

Task: ... (Function to run in container)

"""

import argparse
import boto3
import json
import urllib3

from typing import NamedTuple

PARSER = argparse.ArgumentParser(description="slack alerting")
PARSER.add_argument('-msg', type=str, required=True, default=None, help='slack message to send')
PARSER.add_argument('-pipeline_status', type=str, required=True, default=None, help='pipeline status')
PARSER.add_argument('-aws_region', type=str, required=False, default='eu-central-1', help='AWS region code')
PARSER.add_argument('-slack_channel', type=str, required=False, default='product-ml-ops', help='name of the destined slack channel')
ARGS = PARSER.parse_args()


class SlackAlertingException(Exception):
    """
    Class for handling exceptions for function slack_alerting
    """
    pass


def slack_alerting(msg: str,
                   pipeline_status: str,
                   aws_region: str = 'eu-central-1',
                   slack_channel: str = 'product-ml-ops'
                   ) -> NamedTuple('outputs', [('slack_msg', str),
                                               ('response_status_code', int)
                                               ]
                                   ):
    """
    Send messages and alerts to dedicated Slack channel

    :param msg: str
        Slack messages

    :param pipeline_status: str
        Abbreviated name of the pipeline status
            -> start: Start Kubeflow Pipeline
            -> abort: Abort Kubeflow Pipeline
            -> succeed: End Kubeflow Pipline successfully

    :return: NamedTuple
        Status code of the request
    """
    _client: boto3 = boto3.client('secretsmanager', region_name=aws_region)
    _secret_value: dict = _client.get_secret_value(SecretId=slack_channel)
    _slack_url: str = _secret_value['SecretString'].split('"')[3]
    _http: urllib3.PoolManager = urllib3.PoolManager()
    if pipeline_status == 'abort':
        _slack_msg: dict = {
            "type": "home",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain.text",
                        "text": ":warning: Alarm: Kubeflow Pipeline aborted"
                    }
                },
                {
                    "type": "divider"
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"_{msg}_"
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
                            "text": "Region: *eu-central-1*"
                        }
                    ]
                }
            ]
        }
    elif pipeline_status == 'succeed':
        _slack_msg: dict = {
            "type": "home",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain.text",
                        "text": f":large_green_circle: Kubeflow Pipeline succeeded"
                    }
                },
                {
                    "type": "divider"
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"_{msg}_"
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
    elif pipeline_status == 'start':
        _slack_msg: dict = {
            "type": "home",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain.text",
                        "text": ":warning: Alarm: Kubeflow Pipeline aborted"
                    }
                },
                {
                    "type": "divider"
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"_{msg}_"
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
    else:
        raise SlackAlertingException(f'Pipeline status ({pipeline_status}) not supported')
    _encoded_msg: bytes = json.dumps(_slack_msg).encode('utf-8')
    _response = _http.request(method='POST', url=_slack_url, body=_encoded_msg)
    return [_slack_msg, _response.status]


if __name__ == '__main__':
    slack_alerting(msg=ARGS.msg,
                   pipeline_status=ARGS.pipeline_status,
                   aws_region=ARGS.aws_region,
                   slack_channel=ARGS.slack_channel
                   )
