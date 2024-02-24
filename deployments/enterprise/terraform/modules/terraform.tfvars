environment                                       = "prod"
kubeflow_version                                  = "1.7"
top_level_domain_name                             = "xxx"
domain_name                                       = "xxx"
environment_sub_domain_name                       = "xxx"
namespace_sub_domain_name                         = "xxx"
callback_logout_sub_domain_name                   = "xxx"
cluster_region                                    = "eu-central-1"
cluster_name                                      = "kubeflow"
eks_version                                       = "1.25"
eks_cpu_nodegroup_name                            = "managed-ondemand-cpu"
eks_cpu_node_instance_type                        = "m5.xlarge"
eks_cpu_min_size                                  = 3
eks_cpu_desired_size                              = 3
eks_cpu_max_size                                  = 10
eks_gpu_nodegroup_name                            = "managed-ondemand-gpu"
eks_gpu_node_instance_type                        = null # "p3.16xlarge"
eks_gpu_min_size                                  = 1
eks_gpu_desired_size                              = 3
eks_gpu_max_size                                  = 5
eks_gpu_ami_type                                  = "AL2_x86_64_GPU"
use_cognito                                       = false
cognito_user_pool_client_allowed_oauth_flows      = ["code"]
cognito_user_pool_name                            = "kubeflow-user"
http_header_name                                  = "x-api-key"
http_header_values                                = ["token1", "token2"]
load_balancer_scheme                              = "internet-facing"
ecr_suffix                                        = "ml-ops"
ecr_analytical_data_types                         = "analytical-data-types"
ecr_check_feature_distribution                    = "check-feature-distribution"
ecr_custom_predictor                              = "custom-predictor"
ecr_data_health_check                             = "data-health-check"
ecr_data_typing                                   = "data-typing"
ecr_evolutionary_algorithm                        = "evolutionary-algorithm"
ecr_feature_engineering                           = "feature-engineering"
ecr_feature_selector                              = "feature-selector"
ecr_image_classification_generator                = "image-classification-generator"
ecr_image_processor                               = "image-processor"
ecr_image_translation                             = "image-translation"
ecr_imputation                                    = "imputation"
ecr_interactive_visualizer                        = "interactive-visualizer"
ecr_model_evaluation                              = "model-evaluation"
ecr_model_generator_clustering                    = "model-generator-clustering"
ecr_model_generator_grow_net                      = "model-generator-grow-net"
ecr_model_generator_supervised                    = "model-generator-supervised"
ecr_model_registry                                = "model-registry"
ecr_natural_language_processing                   = "natural-language-processing"
ecr_parallelizer                                  = "parallelizer"
ecr_sampling                                      = "sampling"
ecr_serializer                                    = "serializer"
ecr_slack_alerting                                = "slack-alerting"
ecr_text_classification_generator                 = "text-classification-generator"
use_s3                                            = true
s3_bucket_kubeflow_artifact_store                 = "xxx-artifact-store-prod"
s3_bucket_ml_ops_interim                          = "xxx-ml-ops-interim-prod"
s3_bucket_ml_ops_model_store                      = "xxx-ml-ops-model-store-prod"
s3_bucket_tag_name                                = "ML-Ops interim data"
use_rds                                           = false
db_name                                           = "kubeflow"
db_username                                       = "admin"
db_password                                       = "xxx_xxx_123456"
db_class                                          = "db.m5.large"
db_allocated_storage                              = "20"
db_mysql_engine_version                           = "8.0.34"
db_backup_retention_period                        = 7
db_storage_type                                   = "gp2"
db_deletion_protection                            = false
db_max_allocated_storage                          = 1000
db_publicly_accessible                            = false
db_multi_az                                       = "true"
db_secret_recovery_window_in_days                 = 7
db_tags                                           = {}
pipeline_s3_credential_option                     = "irsa"
use_aws_telemetry                                 = true
notebook_enable_culling                           = true
notebook_cull_idle_time                           = 30
notebook_idleness_check_period                    = 5
grafana_workspace_name                            = "kubeflow"
grafana_workspace_account_access_type             = "CURRENT_ACCOUNT"
grafana_workspace_auth_providers                  = ["AWS_SSO"] # ['SAML']
grafana_workspace_permission_type                 = "SERVICE_MANAGED"
grafana_workspace_data_sources                    = ["CLOUDWATCH", "PROMETHEUS"]
grafana_workspace_notification_destinations       = ["SNS"]
prometheus_workspace_name                         = "kubeflow-prometheus"
cloudwatch_prometheus_log_group_retention_in_days = 365
kf_helm_repo_path                                 = "../../../.."
tags                                              = {}
