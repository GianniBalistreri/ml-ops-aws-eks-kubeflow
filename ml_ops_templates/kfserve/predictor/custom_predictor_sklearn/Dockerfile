FROM python:3.9-alpine

RUN mkdir "/src"

COPY kserve/predictor/custom_predictor_sklearn/src /src
COPY kserve/predictor/custom_predictor_sklearn/pyproject.toml /src

WORKDIR /src

RUN python -m pip install --upgrade pip
RUN python -m pip install setuptools
RUN python -m pip install poetry

RUN poetry config virtualenvs.create false
RUN poetry install --no-dev

ENTRYPOINT ["python", "rest_api.py"]