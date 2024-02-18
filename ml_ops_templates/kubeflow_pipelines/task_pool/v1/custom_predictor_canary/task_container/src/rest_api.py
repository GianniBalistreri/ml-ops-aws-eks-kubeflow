"""

Customized model inference predictor: Sklearn API (Non-Neural Networks)

"""

import argparse
import boto3
import joblib
import numpy as np
import random

from kserve import Model, ModelServer
from ray import serve
from typing import Dict

PARSER = argparse.ArgumentParser(description="restful-api for generate predictions from non-neural network models")
PARSER.add_argument('-model_name', type=str, required=True, default=None, help='name of the pre-trained machine learning model artifact')
PARSER.add_argument('-bucket_name', type=str, required=True, default=None, help='name of the s3 bucket')
PARSER.add_argument('-file_path', type=str, required=True, default=None, help='file path of the model artifact')
PARSER.add_argument('-n_replicas', type=str, required=False, default="1", help='number of replicas to deploy')
PARSER.add_argument('-canary', type=int, required=False, default=0, help='whether to deploy canary deployment or not')
PARSER.add_argument('-canary_first_model_prob', type=float, required=False, default=0.9, help='probability of using first deployed model in canary delpoyment')
PARSER.add_argument('-bucket_name_canary', type=str, required=False, default=None, help='name of the s3 bucket of the second model in canary deployment')
PARSER.add_argument('-file_path_canary', type=str, required=False, default=None, help='file path of the model artifact of the second model in canary deployment')
ARGS = PARSER.parse_args()

MODEL_NAME: str = ARGS.model_name
BUCKET_NAME: str = ARGS.bucket_name
FILE_PATH: str = ARGS.file_path
CANARY: bool = bool(ARGS.canary)


def _handle_canary() -> None:
    pass


def _download_model_artifact(downloaded_model_file_name: str) -> None:
    """
    Download pre-trained model artifact from S3 bucket

    :param downloaded_model_file_name: str
        Name of the uploaded model file
    """
    _s3: boto3 = boto3.client('s3')
    _s3.download_file(BUCKET_NAME, FILE_PATH, downloaded_model_file_name)


def _pre_process(values: list) -> list:
    """
    Pre-processing function

    :return: list
        Pre-processed values
    """
    return values


def _post_process(values: list) -> Dict:
    """
    Post-processing function

    :param values: list
        Predictions

    :return: Dict
        Post-processed predictions
    """
    return {'predictions': values}


class SupervisedMLPredictorException(Exception):
    """
    Class for handling exceptions for class SupervisedMLPredictor
    """
    pass


@serve.deployment(name=MODEL_NAME, num_replicas=int(ARGS.n_replicas))
class SupervisedMLPredictor(Model):
    """
    Class for generating predictions used in inference endpoints of KServe
    """
    def __init__(self):
        self.name: str = MODEL_NAME
        super().__init__(name=self.name)
        self.model = None
        self.load()
        self.ready: bool = True

    def load(self) -> None:
        """
        Load pre-trained machine learning model
        """
        _file_name_model_artifact: str = 'model'
        _download_model_artifact(downloaded_model_file_name=_file_name_model_artifact)
        self.model = joblib.load(filename=_file_name_model_artifact)

    def preprocess(self, inputs: Dict, headers: Dict[str, str] = None) -> Dict:
        """
        Pre-process request data for the machine learning model (Transformer)

        :param inputs: Dict
            Request input

        :param headers: Dict[str, str]
            Request header

        :return: Dict
            Pre-processed data
        """
        return {'instances': [_pre_process(values=instance) for instance in inputs['instances']]}

    async def predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        """
        Generate prediction using pre-trained machine learning model (Predictor)

        :param payload: Dict
            Pre-processed request data

        :param headers: Dict[str, str]
            Request header

        :return: Dict
            Predictions
        """
        _predictions: list = self.model.predict(payload.get('instances'))
        return {"predictions": np.array(_predictions).flatten().tolist()}

    def postprocess(self, inputs: Dict, headers: Dict[str, str] = None) -> Dict:
        """
        Post-process predictions (Transformer)

        :param inputs: Dict
            Predictions

        :param headers: Dict[str, str]
            Request header

        :return: Dict
            Post-process predictions
        """
        return _post_process(values=inputs.get('predictions'))


if __name__ == "__main__":
    ModelServer().start({MODEL_NAME: SupervisedMLPredictor})
