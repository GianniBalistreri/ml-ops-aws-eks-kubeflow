#########
# Misc: #
#########

variable "environment" {
  type        = string
  description = "The name of the environment"
}

variable "tags" {
  type        = map(string)
  description = "Additional tags"
}

########
# EKS: #
########

variable "cluster_name" {
  type        = string
  description = "The name of the EKS cluster"
}

variable "http_header_name" {
  type        = string
  description = "Name of the http header"
}

variable "load_balancer_scheme" {
  type        = string
  description = "Scheme of the load balancer"
}

########
# RDS: #
########

variable "use_rds" {
  type        = bool
  description = "Whether to use RDS for persisting Katib and Pipeline components"
}

variable "security_group_id" {
  type        = string
  description = "SecurityGroup Id of a EKS Worker Node"
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

#######
# S3: #
#######

variable "use_s3" {
  type        = bool
  description = "Whether to use S3 for persisting Kubeflow artefacts"
}

variable "s3_bucket_kubeflow_artifact_store" {
  type        = string
  description = "The name of the Kubeflow artifact store S3 bucket"
}

variable "s3_bucket_ml_ops_interim" {
  type        = string
  description = "The name of the ML-Ops interim S3 bucket"
}

variable "s3_bucket_ml_ops_model_store" {
  type        = string
  description = "The name of the ML-Ops model store S3 bucket"
}

variable "s3_bucket_tag_name" {
  type        = string
  description = "The tag name of the ML-Ops S3 bucket"
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
# Network: #
############

variable "vpc_id" {
  type        = string
  description = "VPC of the EKS cluster"
}

variable "subnet_ids" {
  type        = list(string)
  description = "Subnet ids of the EKS cluster"
}
