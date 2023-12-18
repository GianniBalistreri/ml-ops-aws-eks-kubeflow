FROM python:3.9-alpine

RUN mkdir "/src"

COPY kubeflow_pipelines/task_pool/v1/model_registry/task_container/src /src
COPY kubeflow_pipelines/task_pool/v1/model_registry/task_container/pyproject.toml /src

WORKDIR /src

RUN python -m pip install --upgrade pip
RUN python -m pip install setuptools
RUN python -m pip install poetry

RUN poetry config virtualenvs.create false
RUN poetry install --no-dev

ENTRYPOINT ["python", "task.py"]