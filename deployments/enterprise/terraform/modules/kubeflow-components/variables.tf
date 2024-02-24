#########
# Misc: #
#########

variable "environment" {
  type        = string
  description = "The name of the environment"
}

variable "kf_helm_repo_path" {
  type        = string
  description = "Full path to the location of the helm folder to install from for KF 1.6"
}

variable "addon_context" {
  description = "Input configuration for the addon"
  type        = object({
    aws_caller_identity_account_id = string
    aws_caller_identity_arn        = string
    aws_eks_cluster_endpoint       = string
    aws_partition_id               = string
    aws_region_name                = string
    eks_cluster_id                 = string
    eks_oidc_issuer_url            = string
    eks_oidc_provider_arn          = string
    tags                           = map(string)
    irsa_iam_role_path             = string
    irsa_iam_permissions_boundary  = string
  })
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

########################
# Kubeflow Components: #
########################

variable "use_aws_telemetry" {
  description = "Enable AWS telemetry component"
  type        = bool
}

variable "pipeline_s3_credential_option" {
  description = "The credential type to use to authenticate KFP to use S3. One of [irsa, static]"
  type        = string
  validation {
    condition     = "irsa" == var.pipeline_s3_credential_option || "static" == var.pipeline_s3_credential_option
    error_message = "Must be one of [irsa, static]"
  }
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

variable "callback_logout_sub_domain_name" {
  type        = string
  description = "Name of the callback and logout subdomain"
}

############
# Cognito: #
############

variable "use_cognito" {
  type        = bool
  description = "Whether to use Cognito as authentication tool or dex"
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
# RDS: #
########

variable "kubeflow_db_address" {
  type        = string
  description = "Kubeflow database address"
}

variable "rds_secret" {
  type        = string
  description = "Secret of the RDS database"
}
