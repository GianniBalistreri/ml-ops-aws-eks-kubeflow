provider "aws" {
  alias  = "aws"
  region = var.aws_region
}

provider "aws" {
  alias  = "virginia"
  region = "us-east-1"
}

module "network" {
  source                                       = "./network"
  az_count                                     = var.az_count
  desired_count                                = var.desired_count
  domain_name                                  = var.domain_name
  sub_domain_name                              = var.sub_domain_name
  ecs_enable_dns_hostnames                     = var.ecs_enable_dns_hostnames
  ecs_vpc_cidr_block                           = var.ecs_vpc_cidr_block
  name_prefix                                  = var.name_prefix
  remote_cidr_blocks                           = var.remote_cidr_blocks
  routing_priority                             = var.routing_priority
  vpc_cidr                                     = var.vpc_cidr
  alb_name                                     = var.alb_name
  alb_target_group_name                        = var.alb_target_group_name
  ecs_cluster_name                             = var.ecs_cluster_name
  private_subnet_cidrs                         = var.private_subnet_cidrs
  public_subnet_cidrs                          = var.public_subnet_cidrs
  health_check_path                            = var.health_check_path
  callback_logout_sub_domain_name              = var.callback_logout_sub_domain_name
  cognito_user_pool_client_allowed_oauth_flows = var.cognito_user_pool_client_allowed_oauth_flows
  cognito_user_pool_name                       = var.cognito_user_pool_name
  providers                                    = {
    aws          = aws.aws
    aws.virginia = aws.virginia
  }
}

#module "cognito" {
#  source                                       = "./cognito"
#  cognito_user_pool_name                       = var.cognito_user_pool_name
#  cognito_user_pool_client_allowed_oauth_flows = var.cognito_user_pool_client_allowed_oauth_flows
#  domain_name                                  = var.domain_name
#  sub_domain_name                              = var.sub_domain_name
#  callback_logout_sub_domain_name              = var.callback_logout_sub_domain_name
#  ecs_cluster_name                             = var.ecs_cluster_name
#  alb_dns_name                                 = module.network.alb_dns_name
#  alb_zone_id                                  = module.network.alb_zone_id
#  providers                                    = {
#    aws          = aws.aws
#    aws.virginia = aws.virginia
#  }
#}

module "ecs_fargate" {
  source                                          = "./ecs_fargate"
  autoscaling_max_capacity                        = var.autoscaling_max_capacity
  autoscaling_min_capacity                        = var.autoscaling_min_capacity
  aws_account_id                                  = var.aws_account_id
  aws_access_key_id                               = var.aws_access_key_id
  aws_secret_access_key                           = var.aws_secret_access_key
  aws_region                                      = var.aws_region
  eks_cluster_name                                = var.eks_cluster_name
  ecr_image_tag_mutability                        = var.ecr_image_tag_mutability
  ecr_name                                        = var.ecr_name
  ecr_scan_on_push                                = var.ecr_scan_on_push
  ecs_cluster_container_image                     = var.ecs_cluster_container_image
  ecs_cluster_name                                = var.ecs_cluster_name
  ecs_container_count                             = var.ecs_container_count
  ecs_container_cpu                               = var.ecs_container_cpu
  ecs_container_definitions_awslogs_stream_prefix = var.ecs_container_definitions_awslogs_stream_prefix
  ecs_container_definitions_name                  = var.ecs_container_definitions_name
  ecs_container_memory                            = var.ecs_container_memory
  ecs_container_port                              = var.ecs_container_port
  ecs_fargate_cpu                                 = var.ecs_fargate_cpu
  ecs_fargate_memory                              = var.ecs_fargate_memory
  ecs_fargate_port                                = var.ecs_fargate_port
  ecs_service_name                                = var.ecs_service_name
  ecs_task_definition_cpu                         = var.ecs_task_definition_cpu
  ecs_task_definition_memory                      = var.ecs_task_definition_memory
  ecs_task_definition_name                        = var.ecs_task_definition_name
  ecs_task_definition_network_mode                = var.ecs_task_definition_network_mode
  ecs_task_execution_role_name                    = var.ecs_task_execution_role_name
  ecs_task_role_name                              = var.ecs_task_role_name
  elasticache_cluster_engine                      = var.elasticache_cluster_engine
  elasticache_cluster_id                          = var.elasticache_cluster_id
  elasticache_cluster_node_type                   = var.elasticache_cluster_node_type
  elasticache_cluster_parameter_group_name        = var.elasticache_cluster_parameter_group_name
  elasticache_subnet_group_name                   = var.elasticache_subnet_group_name
  environment                                     = var.environment
  image_url                                       = var.image_url
  lb_path                                         = var.lb_path
  lb_public_access                                = var.lb_public_access
  service_name                                    = var.service_name
  alb_target_group_arn                            = module.network.alb_target_group_arn
  ecs_task_id                                     = module.network.ecs_task_id
  private_subnets_id                              = module.network.private_subnets_id
  public_subnet_cidrs                             = var.public_subnet_cidrs
  subnet_ids                                      = module.network.subnet_ids
  vpc_id                                          = module.network.vpc_id
}
