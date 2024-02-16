FROM python:3.9

RUN mkdir "/src"

COPY ml_ops_templates/kubeflow_pipelines/task_pool/v1/natural_language_processing/task_container/src /src
COPY ml_ops_templates/kubeflow_pipelines/task_pool/v1/natural_language_processing/task_container/pyproject.toml /src

WORKDIR /src

RUN python -m pip install --upgrade pip
RUN python -m pip install setuptools
RUN python -m pip install poetry
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download en_core_web_md
RUN python -m spacy download en_core_web_lg
RUN python -m spacy download de_core_web_sm
RUN python -m spacy download de_core_web_md
RUN python -m spacy download de_core_web_lg

RUN poetry config virtualenvs.create false
RUN poetry install --no-dev

ENTRYPOINT ["python", "task.py"]