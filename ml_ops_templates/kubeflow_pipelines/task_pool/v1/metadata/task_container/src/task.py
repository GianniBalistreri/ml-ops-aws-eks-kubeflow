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
PARSER.add_argument('-output_file_path_date', type=str, required=True, default=None, help='file path of the datetime output')
ARGS = PARSER.parse_args()


def metadata(project_name: str,
             experiment_name: str,
             output_file_path_date: str,
             description: str = None
             ) -> NamedTuple('outputs', [('datetime', str)]):
    """
    Define Kubeflow Pipeline metadata

    :param project_name: str
        Name of the project

    :param experiment_name: str
        Name of the experiment

    :param output_file_path_date: str
        Path of the datime output

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
    with open(output_file_path_date, 'w') as _file:
        json.dump(_metadata.get('date'), _file)
    return [_metadata.get('date')]


if __name__ == '__main__':
    metadata(project_name=ARGS.project_name,
             experiment_name=ARGS.experiment_name,
             output_file_path_date=ARGS.output_file_path_date,
             description=ARGS.description,
             )
