#########
# Misc: #
#########

variable "environment" {
  type        = string
  description = "Name of the environment (prod, staging, dev, etc.)"
}

variable "aws_account_id" {
  type        = string
  description = "AWS account ID"
}

variable "aws_access_key_id" {
  type        = string
  description = "AWS access key ID"
}

variable "aws_secret_access_key" {
  type        = string
  description = "AWS secret access key"
}

variable "aws_region" {
  type        = string
  description = "AWS region code"
}

############
# AWS ECR: #
############

variable "ecr_name" {
  type        = string
  description = "Name of the elastic container registry"
}

variable "ecr_image_tag_mutability" {
  type        = string
  description = "Mutability image tag"
}

variable "ecr_scan_on_push" {
  type        = bool
  description = "Scan container image on push"
}

############
# AWS ECS: #
############

variable "ecs_cluster_container_image" {
  type        = string
  description = "Docker image to run in the ECS cluster"
}

variable "ecs_container_count" {
  type        = string
  description = "Number of docker containers to run"
}

variable "ecs_cluster_name" {
  type        = string
  description = "Name of the elastic cluster service cluster"
}

variable "ecs_service_name" {
  type        = string
  description = "Name of the elastic container service"
}

variable "ecs_task_definition_name" {
  type        = string
  description = "Name of the elastic container service task definition"
}

variable "ecs_task_definition_network_mode" {
  type        = string
  description = "Mode of the elastic container service task definition"
}

variable "ecs_task_definition_cpu" {
  type        = string
  description = "Number of CPU's of the elastic container service task definition"
}

variable "ecs_task_definition_memory" {
  type        = string
  description = "Amount of memory of the elastic container service task definition"
}

variable "ecs_task_role_name" {
  type        = string
  description = "Name of the elastic container service role"
}

variable "ecs_container_definitions_name" {
  type        = string
  description = "Name of the elastic container service container definitions"
}

variable "ecs_container_definitions_awslogs_stream_prefix" {
  type        = string
  description = "Name of the elastic container service container definitions aws logs stream prefix"
}

variable "ecs_task_execution_role_name" {
  type        = string
  description = "ECS task execution role name"
}

variable "ecs_fargate_port" {
  type        = number
  description = "Port exposed by the docker image to redirect traffic to"
}

variable "ecs_fargate_cpu" {
  type        = number
  description = "Fargate instance CPU units to provision (1 vCPU = 1024 CPU units)"
}

variable "ecs_fargate_memory" {
  type        = number
  description = "Fargate instance memory to provision (in MiB)"
}

variable "ecs_container_port" {
  type        = number
  description = "What port number the application inside the docker container is binding to"
}

variable "ecs_container_cpu" {
  type        = number
  description = "How much CPU to give the container. 1024 is 1 CPU"
}

variable "ecs_container_memory" {
  type        = number
  description = "How much memory in megabytes to give the container"
}

variable "service_name" {
  type        = string
  description = "A name for the service"
}

variable "image_url" {
  type        = string
  description = "the url of a docker image that contains the application process that will handle the traffic for this service"
}

variable "name" {
  description = "Name given resources"
  type        = string
  default     = "aws-ia"
}

variable "lb_public_access" {
  type        = bool
  description = "Make LB accessible publicly"
}

variable "lb_path" {
  type        = string
  description = "A path on the public load balancer that this service should be connected to. Use * to send all load balancer traffic to this service."
}

variable "autoscaling_min_capacity" {
  type        = number
  description = "Minimum number of cluster EC2 instances"
}

variable "autoscaling_max_capacity" {
  type        = number
  description = "Maximum number of cluster EC2 instances"
}

variable "elasticache_cluster_id" {
  type        = string
  description = "Name of the elasticache cluster"
}

variable "elasticache_cluster_engine" {
  type        = string
  description = "Name of the elasticache cluster engine"
}

variable "elasticache_cluster_node_type" {
  type        = string
  description = "Name of the elasticache cluster node type"
}

variable "elasticache_cluster_parameter_group_name" {
  type        = string
  description = "Name of the elasticache cluster parameter group"
}

variable "elasticache_subnet_group_name" {
  type        = string
  description = "Name of the elasticache cluster"
}

################
# AWS Network: #
################

variable "vpc_id" {
  type        = string
  description = "ECS VPC ID"
}

variable "private_subnets_id" {
  type        = list(string)
  description = "Private subnets id"
}

variable "public_subnet_cidrs" {
  type        = list(string)
  description = "Public subnet CIDRs"
}

variable "subnet_ids" {
  type        = list(string)
  description = "Subnet ids"
}

##################################
# AWS Application Load Balancer: #
##################################

variable "ecs_task_id" {
  type        = string
  description = "ECS task id for the security group"
}

variable "alb_target_group_arn" {
  type        = string
  description = "ARN of the ALB target group"
}
