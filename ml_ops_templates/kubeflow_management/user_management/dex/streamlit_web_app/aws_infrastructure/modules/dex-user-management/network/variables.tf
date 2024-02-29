################
# AWS Network: #
################

variable "domain_name" {
  type        = string
  description = "Name of the domain"
}

variable "sub_domain_name" {
  type        = string
  description = "Name of the sub domain"
}

variable "vpc_cidr" {
  type        = string
  description = "The CIDR block of the main vpc"
}

variable "remote_cidr_blocks" {
  type        = list(any)
  description = "By default cidr_blocks are locked down. (Update to 0.0.0.0/0 if full public access is needed)"
}

variable "az_count" {
  type        = string
  description = "Number of AZs to cover in a given region"
}

variable "routing_priority" {
  type        = number
  description = "The priority for the routing rule added to the load balancer. This only applies if your have multiple services which have been assigned to different paths on the load balancer."
}

variable "desired_count" {
  type        = number
  description = "How many copies of the service task to run"
}

variable "name_prefix" {
  type        = string
  description = "Name Prefix"
}

variable "subnet_tag" {
  description = "Tags used to filter ecs subnets "
  type        = string
  default     = "ecs-subnets"
}

variable "ecs_vpc_cidr_block" {
  type        = string
  description = "VPC port CIDR block"
}

variable "ecs_enable_dns_hostnames" {
  type        = bool
  description = "Whether to enable dns hostnames or not"
}

variable "public_subnet_cidrs" {
  type        = list(string)
  description = "Public subnet CIDRs"
}

variable "private_subnet_cidrs" {
  type        = list(string)
  description = "Private subnet CIDRs"
}

##################################
# AWS Application Load Balancer: #
##################################

variable "ecs_cluster_name" {
  type        = string
  description = "Name of the elastic cluster service cluster"
}

variable "health_check_path" {
  type        = string
  description = "Path of the health check endpoint"
}

variable "alb_name" {
  type        = string
  description = "Name of the application load balancer"
}

variable "alb_target_group_name" {
  type        = string
  description = "Name of the application load balancer target group"
}
