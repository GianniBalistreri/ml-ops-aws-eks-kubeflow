variable "tags" {
  description = "Additional tags"
  type        = map(string)
  default     = {}
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
