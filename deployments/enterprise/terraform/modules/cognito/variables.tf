#########
# Misc: #
#########

variable "addon_context" {
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
  description = "Input configuration for the addon"
}

variable "tags" {
  description = "Additional tags"
  type        = map(string)
  default     = {}
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

variable "load_balancer_scheme" {
  type        = string
  description = "Load Balancer Scheme"
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
# EKS: #
########

variable "cluster_name" {
  type        = string
  description = "EKS cluster name"
}
