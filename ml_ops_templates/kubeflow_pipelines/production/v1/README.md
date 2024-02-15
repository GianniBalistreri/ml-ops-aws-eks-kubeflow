# Kubeflow Pipeline Utils

Utility functions for easy and effective deployment of Kubeflow Pipelines.

## 1.) Task pool functions:

Define task pool functions for each deployed docker image application.

    - analytical_data_types
    - check_feature_distribution
    - data_health_check
    - data_typing
    - evolutionary_algorithm
    - feature_engineering
    - feature_selector
    - image_processor
    - image_translation
    - imputation
    - interactive_visualizer
    - model_evaluation
    - model_generator_clustering
    - model_generator_supervised
    - model_registry
    - parallelizer
    - sampling
    - serializer
    - slack_alerting

## 2.) Set ContainerOp parameters:

Define resource attribution for each pipeline component

    - Number of CPU request
    - Number of CPU limit
    - Number of GPU request
    - Name of the GPU vendor
    - Memory request
    - Memory limit
    - Ephemeral (temporary) storage request
    - Ephemeral (temporary) storage limit
    - Name of the instance
    - Maximum days of component caching

## 3.) Dex authentications

Receive session cookies to authenticate from outside kubernetes cluster.
