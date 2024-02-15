"""

Customized model inference predictor: Sklearn API (Non-Neural Networks)

"""

import argparse
import boto3
import joblib
import numpy as np

from kserve import Model, ModelServer
from typing import Dict

PARSER = argparse.ArgumentParser(description="restful-api for generate predictions from non-neural network models")
PARSER.add_argument('-model_name', type=str, required=True, default=None, help='name of the pre-trained machine learning model artifact')
PARSER.add_argument('-bucket_name', type=str, required=False, default='shopware-ml-ops-model-store-dev', help='name of the s3 bucket')
PARSER.add_argument('-file_path', type=str, required=False, default='occupancy/occupancy_xgb_model.joblib', help='file path of the model artifact')
ARGS = PARSER.parse_args()


def _download_model_artifact(bucket_name: str, file_path: str, downloaded_model_file_name: str) -> None:
    """
    Download pre-trained model artifact from S3 bucket

    :param bucket_name: str
        Name of the S3 bucket

    :param file_path: str
        File path of the model artifact

    :param downloaded_model_file_name: str
        Name of the downloaded model file
    """
    _s3: boto3 = boto3.client('s3')
    _s3.download_file(bucket_name, file_path, downloaded_model_file_name)


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


class SupervisedMLPredictor(Model):
    """
    Class for generating predictions used in inference endpoints of KServe
    """
    def __init__(self, name: str, bucket_name: str, file_path: str):
        super().__init__(name)
        self.name: str = name
        self.bucket_name: str = bucket_name
        self.file_path: str = file_path
        self.model = None
        self.load()
        self.ready: bool = True

    def load(self) -> None:
        """
        Load pre-trained machine learning model
        """
        _file_name_model_artifact: str = 'model'
        _download_model_artifact(bucket_name=self.bucket_name,
                                 file_path=self.file_path,
                                 downloaded_model_file_name=_file_name_model_artifact
                                 )
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

    def predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
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
    model = SupervisedMLPredictor(name=ARGS.model_name,
                                  bucket_name=ARGS.bucket_name,
                                  file_path=ARGS.file_path
                                  )
    ModelServer().start([model])
