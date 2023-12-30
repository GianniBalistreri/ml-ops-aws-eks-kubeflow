"""

Customized model inference predictor: Sklearn API (Non-Neural Networks)

"""

import argparse
import numpy as np
import torch

from kserve import Model, ModelServer
from torchvision import models
from typing import Dict


class SupervisedMLPredictor(Model):
    """
    Class for generating predictions used in inference endpoints of KServe
    """
    def __init__(self, name: str):
        super().__init__(name)
        self.name: str = name
        self.model = None
        self.load()

    def load(self):
        self.model = models.alexnet(pretrained=True)
        self.model.eval()

    def preprocess(self, inputs: Dict, headers: Dict[str, str] = None) -> Dict:
        return {'instances': [image_transform(instance) for instance in inputs['instances']]}

    def predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        img_data = payload["instances"][0]["image"]["b64"]
        raw_img_data = base64.b64decode(img_data)
        input_image = Image.open(io.BytesIO(raw_img_data))
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image).unsqueeze(0)
        output = self.model(input_tensor)
        torch.nn.functional.softmax(output, dim=1)
        values, top_5 = torch.topk(output, 5)
        result = values.flatten().tolist()
        response_id = generate_uuid()
        return {"predictions": result}

    def postprocess(self, inputs: Dict, headers: Dict[str, str] = None) -> Dict:
        return inputs


if __name__ == "__main__":
    model = Predictor("custom-model")
    ModelServer().start([model])
