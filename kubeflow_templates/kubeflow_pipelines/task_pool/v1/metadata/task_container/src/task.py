"""

Task: ... (Function to run in container)

"""

import argparse
import json

from datetime import datetime
from typing import NamedTuple


PARSER = argparse.ArgumentParser(description="define metadata")
PARSER.add_argument('-project_name', type=str, required=False, default=None, help='name of the project')
PARSER.add_argument('-experiment_name', type=str, required=False, default=None, help='name of the experiment')
PARSER.add_argument('-description', type=str, required=False, default=None, help='description')
ARGS = PARSER.parse_args()


def metadata(project_name: str,
             experiment_name: str,
             description: str = None
             ) -> NamedTuple('outputs', [('metadata', dict)]):
    """
    Define Kubeflow Pipeline metadata

    :param project_name: str
        Name of the project

    :param experiment_name: str
        Name of the experiment

    :param description: str
        Description

    :return: NamedTuple
        Kubeflow pipeline metadata
    """
    _metadata: dict = dict(project_name=project_name,
                           experiment_name=experiment_name,
                           description=description,
                           date=str(datetime.now()).replace(' ', '-').replace(':', '-').replace('.', '-')
                           )
    with open('metadata.json', 'w') as _file:
        json.dump(_metadata, _file)
    return [_metadata]


if __name__ == '__main__':
    metadata(project_name=ARGS.project_name,
             experiment_name=ARGS.experiment_name,
             description=ARGS.description,
             )
