#########
# Misc: #
#########

variable "environment" {
  type        = string
  description = "The name of the environment"
}

variable "kubeflow_version" {
  type        = string
  description = "Kubeflow version"
}

variable "kf_helm_repo_path" {
  type        = string
  description = "Full path to the location of the helm folder to install from for KF 1.6"
}

variable "tags" {
  type        = map(string)
  description = "Additional tags"
}

#######
# S3: #
#######

variable "s3_bucket_ml_ops_interim" {
  type        = string
  description = "The name of the ML-Ops interim S3 bucket"
}

variable "s3_bucket_kubeflow_artifact_store" {
  type        = string
  description = "The name of the Kubeflow artifact store S3 bucket"
}

variable "s3_bucket_ml_ops_model_store" {
  type        = string
  description = "The name of the ML-Ops model store S3 bucket"
}

variable "s3_bucket_tag_name" {
  type        = string
  description = "The tag name of the ML-Ops S3 bucket"
}

########################
# Kubeflow Components: #
########################

variable "use_s3" {
  type        = bool
  description = "Whether to use S3 for persisting Kubeflow artefacts"
}

variable "use_aws_telemetry" {
  type        = bool
  description = "Enable AWS telemetry component"
}

variable "pipeline_s3_credential_option" {
  type = string
  validation {
    condition     = "irsa" == var.pipeline_s3_credential_option || "static" == var.pipeline_s3_credential_option
    error_message = "Must be one of [irsa, static]"
  }
  description = "The credential type to use to authenticate KFP to use S3. One of [irsa, static]"
}

variable "notebook_enable_culling" {
  type        = string
  description = "Enable Notebook culling feature. If set to true then the Notebook Controller will scale all Notebooks with Last activity older than the notebook_cull_idle_time to zero"
}

variable "notebook_cull_idle_time" {
  type        = string
  description = "If a Notebook's LAST_ACTIVITY_ANNOTATION from the current timestamp exceeds this value then the Notebook will be scaled to zero (culled). ENABLE_CULLING must be set to 'true' for this setting to take effect.(minutes)"
}

variable "notebook_idleness_check_period" {
  type        = string
  description = "How frequently the controller should poll each Notebook to update its LAST_ACTIVITY_ANNOTATION (minutes)"
}

########
# RDS: #
########

variable "use_rds" {
  type        = bool
  description = "Whether to use RDS for persisting Katib and Pipeline components"
}

variable "db_name" {
  type        = string
  description = "Database name"
}

variable "db_username" {
  type        = string
  description = "Database admin account username"
}

variable "db_password" {
  type        = string
  description = "Database admin account password"
}

variable "db_class" {
  type        = string
  description = "Database instance type"
}

variable "db_allocated_storage" {
  type        = string
  description = "The size of the database (Gb)"
}

variable "db_mysql_engine_version" {
  type        = string
  description = "The engine version of MySQL"
}

variable "db_backup_retention_period" {
  type        = number
  description = "Number of days to retain backups for"
}

variable "db_storage_type" {
  type        = string
  description = "Instance storage type: standard, gp2, or io1"
}

variable "db_deletion_protection" {
  type        = bool
  description = "Prevents the deletion of the instance when set to true"
}

variable "db_max_allocated_storage" {
  type        = number
  description = "The upper limit of scalable storage (Gb)"
}

variable "db_publicly_accessible" {
  type        = bool
  description = "Makes the instance publicly accessible when true"
}

variable "db_multi_az" {
  type        = string
  description = "Enables multi AZ for the master database"
}

variable "db_secret_recovery_window_in_days" {
  type        = number
  description = "Recovery window in days of the Database secret"
}

variable "db_tags" {
  type        = map(string)
  description = "Additional tags"
}

############
# Network: #
############

variable "top_level_domain_name" {
  type        = string
  description = "Name of the top level domain"
}

variable "domain_name" {
  type        = string
  description = "Name of the domain"
}

variable "environment_sub_domain_name" {
  type        = string
  description = "Name of the environment subdomain"
}

variable "namespace_sub_domain_name" {
  type        = string
  description = "Name of the namespace subdomain"
}

variable "http_header_name" {
  type        = string
  description = "Name of the http header"
}

variable "http_header_values" {
  type        = list(string)
  description = "Values if the http header"
}

variable "load_balancer_scheme" {
  type        = string
  description = "Scheme of the load balancer"
}

########
# EKS: #
########

variable "cluster_name" {
  type = string
  validation {
    condition     = length(var.cluster_name) > 0 && length(var.cluster_name) <= 19
    error_message = "The cluster name must be between [1, 19] characters"
  }
  description = "Name of cluster"
}

variable "cluster_region" {
  type        = string
  description = "Region to create the cluster"
}

variable "eks_version" {
  type        = string
  description = "The EKS version to use"
}

variable "eks_cpu_nodegroup_name" {
  type        = string
  description = "Name of the EKS CPU nodegroup"
}

variable "eks_cpu_min_size" {
  type        = number
  description = "Minimum number of CPU instances used for EKS"
}

variable "eks_cpu_desired_size" {
  type        = number
  description = "Desired number of CPU instances used for EKS"
}

variable "eks_cpu_max_size" {
  type        = number
  description = "Maximum number of CPU instances used for EKS"
}

variable "eks_cpu_node_instance_type" {
  type        = string
  description = "The instance type of an EKS node"
}

variable "eks_gpu_nodegroup_name" {
  type        = string
  description = "Name of the EKS GPU nodegroup"
}

variable "eks_gpu_min_size" {
  type        = number
  description = "Minimum number of GPU instances used for EKS"
}

variable "eks_gpu_desired_size" {
  type        = number
  description = "Desired number of GPU instances used for EKS"
}

variable "eks_gpu_max_size" {
  type        = number
  description = "Maximum number of GPU instances used for EKS"
}

variable "eks_gpu_node_instance_type" {
  type        = string
  description = "The instance type of a gpu EKS node. Will result in the creation of a separate gpu node group when not null"
}

variable "eks_gpu_ami_type" {
  type        = string
  description = "Name of the ami type"
}

########
# ECR: #
########

variable "ecr_suffix" {
  type        = string
  description = "The suffix of the Elastic Container Registry (ECR) repositories"
}

variable "ecr_analytical_data_types" {
  type        = string
  description = "The name of the analytical data types Elastic Container Registry (ECR) repository"
}

variable "ecr_anomaly_detection" {
  type        = string
  description = "The name of the anomaly detection Elastic Container Registry (ECR) repository"
}

variable "ecr_data_health_check" {
  type        = string
  description = "The name of the data health check Elastic Container Registry (ECR) repository"
}

variable "ecr_evolutionary_algorithm" {
  type        = string
  description = "The name of the evolutionary algorithm Elastic Container Registry (ECR) repository"
}

variable "ecr_feature_engineering" {
  type        = string
  description = "The name of the feature engineering Elastic Container Registry (ECR) repository"
}

variable "ecr_feature_selector" {
  type        = string
  description = "The name of the feature selector Elastic Container Registry (ECR) repository"
}

variable "ecr_model_evaluation" {
  type        = string
  description = "The name of the model evaluation Elastic Container Registry (ECR) repository"
}

variable "ecr_model_generator_clustering" {
  type        = string
  description = "The name of the model generator clustering Elastic Container Registry (ECR) repository"
}

variable "ecr_model_generator_grow_net" {
  type        = string
  description = "The name of the model generator grow net Elastic Container Registry (ECR) repository"
}

variable "ecr_model_generator_supervised" {
  type        = string
  description = "The name of the model generator supervised Elastic Container Registry (ECR) repository"
}

variable "ecr_natural_language_processing" {
  type        = string
  description = "The name of the natural language processing Elastic Container Registry (ECR) repository"
}

variable "ecr_sampling" {
  type        = string
  description = "The name of the sampling Elastic Container Registry (ECR) repository"
}

variable "ecr_slack_alerting" {
  type        = string
  description = "The name of the slack alerting Elastic Container Registry (ECR) repository"
}

############
# Cognito: #
############

variable "use_cognito" {
  type        = bool
  description = "Whether to use AWS Cognito as user management tool or not"
}

variable "cognito_user_pool_client_allowed_oauth_flows" {
  type        = list(string)
  description = "Allowed Oauth flows for cognito user pool client"
}

variable "cognito_user_pool_name" {
  type        = string
  description = "Cognito User Pool name"
}

########
# AMG: #
########

variable "grafana_workspace_name" {
  type        = string
  description = "Name of the grafana workspace"
}

variable "grafana_workspace_account_access_type" {
  type        = string
  description = "Name of the account access type used in grafana workspace"
}

variable "grafana_workspace_auth_providers" {
  type        = list(string)
  description = "Names of the authentication providers used in grafana workspace"
}

variable "grafana_workspace_permission_type" {
  type        = string
  description = "Name of the permission type used in grafana workspace"
}

variable "grafana_workspace_data_sources" {
  type        = list(string)
  description = "Names of the data sources used in grafana workspace"
}

variable "grafana_workspace_notification_destinations" {
  type        = list(string)
  description = "Names of the notification destinations used in grafana workspace"
}

########
# AMP: #
########

variable "prometheus_workspace_name" {
  type        = string
  description = "Name of the prometheus workspace"
}

variable "cloudwatch_prometheus_log_group_retention_in_days" {
  type        = number
  description = "Retention of the cloudwatch prometheus log group in days"
}
