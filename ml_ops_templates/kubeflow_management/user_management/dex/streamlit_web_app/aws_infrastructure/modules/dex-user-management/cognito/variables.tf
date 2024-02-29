############
# Network: #
############

variable "domain_name" {
  type        = string
  description = "Name of the domain"
}

variable "sub_domain_name" {
  type        = string
  description = "Name of the sub domain"
}

variable "callback_logout_sub_domain_name" {
  type        = string
  description = "Name of the callback and logout subdomain"
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
# ECS: #
########

variable "ecs_cluster_name" {
  type        = string
  description = "ECS cluster name"
}


########
# ALB: #
########

variable "alb_dns_name" {
  type        = string
  description = "DNS name of the application load balancer"
}

variable "alb_zone_id" {
  type        = string
  description = "Zone ID of the application load balancer"
}
