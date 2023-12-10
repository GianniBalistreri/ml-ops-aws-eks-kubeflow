variable "tags" {
  description = "Additional tags (e.g. `map('BusinessUnit`,`XYZ`)"
  type        = map(string)
  default     = {}
}

variable "domain_name" {
  type        = string
  description = "Name of the domain"
}

variable "sub_domain_name" {
  type        = string
  description = "Name of the subdomain"
}