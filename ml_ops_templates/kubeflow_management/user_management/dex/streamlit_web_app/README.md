# Kubeflow User Management

Handle registration of Kubeflow users using Dex as an authentication provider efficiently.

1.) AWS Services:

The technical infrastructure contain the following AWS services:

- Cognito for handling user authentication
- ECS for application deployment
- Fargate for handling EC2 instances

2.) Application:


3.) Local Deployment:

To use the app locally the following steps must be carried out:

- install dependencies listed in: "pyproject.toml"
- define environment variables:
  - AWS_ACCESS_KEY_ID=<>
  - AWS_SECRET_ACCESS_KEY=<>
  - AWS_ACCOUNT_ID=<>
  - AWS_REGION=<>
  - CLUSTER_NAME=<>
- go to directory: "ml_ops_templates/kubeflow_management/user_management/dex/streamlit_web_app/app/src"
- run application using command: streamlit run app.py

