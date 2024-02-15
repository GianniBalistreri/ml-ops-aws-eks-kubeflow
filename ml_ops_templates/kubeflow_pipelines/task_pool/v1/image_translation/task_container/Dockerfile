FROM python:3.9

RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx libglib2.0-0

RUN mkdir "/src"

COPY kubeflow_pipelines/task_pool/v1/image_translation/task_container/src /src
COPY kubeflow_pipelines/task_pool/v1/image_translation/task_container/pyproject.toml /src

WORKDIR /src

RUN python -m pip install --upgrade pip
RUN python -m pip install setuptools
RUN python -m pip install poetry
RUn python -m pip install git+https://www.github.com/keras-team/keras-contrib.git

RUN poetry config virtualenvs.create false
RUN poetry install --no-dev

ENTRYPOINT ["python", "task.py"]