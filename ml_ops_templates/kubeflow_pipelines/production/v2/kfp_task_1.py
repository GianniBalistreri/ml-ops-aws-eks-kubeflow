"""
Kubeflow Pipeline Task: kfp v2
"""

import os

from task import add
from kfp.dsl import component
from typing import NamedTuple

@component(
    base_image='python:3.9',
    target_image=f'{os.getenv("AWS_ACCOUNT_ID")}.dkr.ecr.{os.getenv("AWS_ACCOUNT_REGION")}.amazonaws.com/{os.getenv("IMAGE_NAME")}:{os.getenv("IMAGE_TAG")}',
    packages_to_install=[],
    pip_index_urls=None,
    output_component_file='task_component.yaml',
    install_kfp_package=True,
    kfp_package_path=None
)
def task_component(a: float, b: float) -> NamedTuple('outputs', [('sum', float)]):
    return add(a=a, b=b)
