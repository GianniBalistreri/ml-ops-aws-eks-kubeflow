FROM python:3.9

RUN mkdir "/src"

COPY kubeflow_pipelines/task_pool/v1/custom_predictor_test/task_container/src /src
COPY kubeflow_pipelines/task_pool/v1/custom_predictor_test/task_container/pyproject.toml /src

WORKDIR /src

RUN apt-get update && apt-get install -y gcc \
                                        libc-dev

RUN python -m pip install --upgrade pip
RUN python -m pip install setuptools
RUN python -m pip install poetry

RUN poetry config virtualenvs.create false
RUN poetry install --no-dev

ENTRYPOINT ["python", "rest_api.py"]