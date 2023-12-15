FROM python:3.8-alpine

RUN mkdir "/src"

COPY kubeflow_pipelines/task_pool/v1/interactive_visualizer/task_container/src /src
COPY kubeflow_pipelines/task_pool/v1/interactive_visualizer/task_container/pyproject.toml /src

WORKDIR /src

RUN python -m pip install --upgrade pip
RUN python -m pip install setuptools
RUN python -m pip install poetry

RUN poetry config virtualenvs.create false
RUN poetry install --no-dev

ENTRYPOINT ["python", "task.py"]