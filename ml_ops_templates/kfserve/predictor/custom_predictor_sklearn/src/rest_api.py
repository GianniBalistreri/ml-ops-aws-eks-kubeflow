"""

Customized model inference predictor: Sklearn API (Non-Neural Networks)

"""

import argparse
import numpy as np

from joblib import load
from kserve import Model, ModelServer
from typing import Dict


PARSER = argparse.ArgumentParser(description="restful-api for generate predictions")
PARSER.add_argument('-model_name', type=str, required=True, default=None, help='name of the pre-trained machine learning model artifact')
ARGS = PARSER.parse_args()


def pre_process(values: list) -> list:
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


class SupervisedMLPredictor(Model):
    """
    Class for generating predictions used in inference endpoints of KServe
    """
    def __init__(self, name: str):
        super().__init__(name)
        self.name: str = name
        self.model = None
        self.load()

    def load(self) -> None:
        """
        Load pre-trained machine learning model
        """
        self.model = load(filename=self.name)

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
        return {'instances': [pre_process(values=instance) for instance in inputs['instances']]}

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
    model = SupervisedMLPredictor(name=ARGS.model_name)
    ModelServer().start([model])
