#FROM python:3.9
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

RUN apt-get update --yes --quiet && DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    software-properties-common \
    build-essential apt-utils \
    wget curl vim git ca-certificates kmod \
    nvidia-driver-525 \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update --yes --quiet
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3.10-lib2to3 \
    python3.10-gdbm \
    python3.10-tk \
    pip

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 999 \
    && update-alternatives --config python3 && ln -s /usr/bin/python3 /usr/bin/python

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

##############

RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx libglib2.0-0

RUN mkdir "/src"

COPY kubeflow_pipelines/task_pool/v1/image_translation_gpu/task_container/src /src
COPY kubeflow_pipelines/task_pool/v1/image_translation_gpu/task_container/pyproject.toml /src

WORKDIR /src

RUN python -m pip install --upgrade pip
RUN python -m pip install setuptools
RUN python -m pip install tensorflow[and-cuda]
RUN python -m pip install poetry
RUn python -m pip install git+https://www.github.com/keras-team/keras-contrib.git

RUN poetry config virtualenvs.create false
RUN poetry install --no-dev

ENTRYPOINT ["python", "task.py"]