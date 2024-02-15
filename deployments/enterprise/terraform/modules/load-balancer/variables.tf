############
# Network: #
############

variable "http_header_name" {
  type        = string
  description = "Name of the http header"
}

variable "load_balancer_scheme" {
  type        = string
  description = "Scheme of the load balancer"
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

########
# EKS: #
########

variable "cluster_name" {
  description = "Name of cluster"
  type        = string

  validation {
    condition     = length(var.cluster_name) > 0 && length(var.cluster_name) <= 19
    error_message = "The cluster name must be between [1, 19] characters"
  }
}
