FROM python:3.9

RUN mkdir "/src"

COPY kubeflow_pipelines/task_pool/v1/check_for_retraining/structured_data/check_feature_distribution/task_container/src /src
COPY kubeflow_pipelines/task_pool/v1/check_for_retraining/structured_data/check_feature_distribution/task_container/pyproject.toml /src

WORKDIR /src

RUN python -m pip install --upgrade pip
RUN python -m pip install setuptools
RUN python -m pip install poetry

RUN poetry config virtualenvs.create false
RUN poetry install --no-dev

ENTRYPOINT ["python", "task.py"]