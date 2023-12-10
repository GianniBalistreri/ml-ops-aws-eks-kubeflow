########
# EKS: #
########

variable "cluster_region" {
  type        = string
  description = "Region of the deployed EKS"
}

variable "cluster_name" {
  type        = string
  description = "Name of the deployed EKS cluster"
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

###############
# Cloudwatch: #
###############

variable "cloudwatch_prometheus_log_group_retention_in_days" {
  type        = number
  description = "Retention of the cloudwatch prometheus log group in days"
}
