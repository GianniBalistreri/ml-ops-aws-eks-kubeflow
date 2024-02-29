include {
  path = find_in_parent_folders()
}

terraform {
  source = "../../../modules/dex-user-management"
}

inputs = {
  aws_account_id                                  = "xxx"
  aws_region                                      = "xxx"
  terraform_role_arn_name                         = "TerraformServiceRole"
  s3_bucket_name_terraform_backend                = "xxx-kubeflow-user-management-tfstate-production"
  terraform_backend_key                           = "dex-user-management/terraform.tfstate"
  environment                                     = "prod"
  domain_name                                     = "xxx" #"bla.io"
  sub_domain_name                                 = "xxx"
  callback_logout_sub_domain_name                 = "xxx"
  vpc_id                                          = "vpc"
  vpc_cidr                                        = "10.0.0.0/16"
  remote_cidr_blocks                              = ["10.0.0.0/32"]
  public_subnet_cidrs                             = ["10.0.4.0/24", "10.0.5.0/24", "10.0.6.0/24"]
  private_subnet_cidrs                            = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  az_count                                        = "3"
  routing_priority                                = 1
  desired_count                                   = 2
  name_prefix                                     = "fw"
  subnet_tag                                      = "ecs-subnets"
  ecs_vpc_cidr_block                              = "10.0.0.0/16"
  ecs_enable_dns_hostnames                        = true
  ecr_name                                        = "kubeflow-user-management"
  ecr_image_tag_mutability                        = "MUTABLE"
  ecr_scan_on_push                                = true
  ecs_cluster_container_image                     = "latest"
  ecs_container_count                             = "1"
  ecs_cluster_name                                = "kubeflow-user-management-cluster"
  ecs_service_name                                = "kubeflow-user-management"
  ecs_task_definition_name                        = "kubeflow-user-management-task"
  ecs_task_definition_network_mode                = "awsvpc"
  ecs_task_definition_cpu                         = "4"
  ecs_task_definition_memory                      = "64"
  ecs_task_role_name                              = "kubeflow-user-management"
  ecs_container_definitions_name                  = "kubeflow-user-management"
  ecs_container_definitions_awslogs_stream_prefix = "kubeflow-user-management-ecs"
  ecs_task_execution_role_name                    = "KubeflowUserManagementEcsTaskExecutionRole"
  ecs_fargate_port                                = 80
  ecs_fargate_cpu                                 = 256
  ecs_fargate_memory                              = 512
  ecs_container_port                              = 80
  ecs_container_cpu                               = 64
  ecs_container_memory                            = 256
  service_name                                    = "nginx"
  image_url                                       = "nginx"
  health_check_path                               = "/healthcheck"
  lb_public_access                                = true
  lb_path                                         = "*"
  alb_name                                        = "kubeflow-user-management"
  alb_target_group_name                           = "kubeflow-user-management"
  cognito_user_pool_name                          = "kubeflow-user-management"
  cognito_user_pool_client_allowed_oauth_flows    = ["code"]
  autoscaling_min_capacity                        = 1
  autoscaling_max_capacity                        = 2
  elasticache_cluster_id                          = "kubeflow-user-management"
  elasticache_cluster_engine                      = "redis"
  elasticache_cluster_node_type                   = "cache.t4g.micro"
  elasticache_cluster_parameter_group_name        = "default.redis7"
  elasticache_subnet_group_name                   = "kubeflow-user-management"
  name                                            = "kubeflow-user-management"
}
