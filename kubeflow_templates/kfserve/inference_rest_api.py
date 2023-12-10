"""
REST-API Inference Service: Realtime
"""

import boto3
import requests
import json

# API-Endpoint of the pre-trained machine learning model artefact:
api_endpoint: str = "http://pytorch-classifier.default.example.com/v1/models/pytorch-classifier:predict"
headers: dict = {"Content-Type": "application/json"}

# Simple Notification Service (SNS) for sending predictions back to Shopware:
sns_client: boto3 = boto3.client('sns', region_name='eu-central-1')
topic_arn: str = 'arn:aws:sns:your-region:your-account-id:your-topic-name'

# Kinesis Data Stream containing data:
stream_name: str = 'your-data-stream-name'
kinesis_client: boto3 = boto3.client('kinesis', region_name='eu-central-1')

# Extract data from Kinesis Data Stream:
response_kinesis: json = kinesis_client.get_records(StreamName=stream_name, Limit=100)
for record in response_kinesis['Records']:
    data = json.loads(record['Data'])
    # Send data to REST-API model endpoint:
    response_model: json = requests.post(api_endpoint, json=data, headers=headers)
    if response_model.status_code == 200:
        # Send prediction back to Shopware:
        response_message: json = sns_client.publish(TopicArn=topic_arn, Message=response_model.json())
        print("Nachricht gesendet:", response_message['MessageId'])
    else:
        print("Fehler beim Senden der Daten")
