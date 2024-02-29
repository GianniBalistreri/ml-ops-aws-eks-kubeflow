FROM python:3.9

RUN mkdir "/src"

COPY ml_ops_templates/kubeflow_management/dex/streamlit_web_app/app/src /src
COPY ml_ops_templates/kubeflow_management/dex/streamlit_web_app/app/pyproject.toml /src

WORKDIR /src

RUN python -m pip install --upgrade pip
RUN python -m pip install setuptools
RUN python -m pip install poetry

RUN poetry config virtualenvs.create false
RUN poetry install --no-dev

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]