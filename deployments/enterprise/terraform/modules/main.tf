data "aws_ec2_instance_type_offerings" "availability_zones_cpu" {
  filter {
    name   = "instance-type"
    values = [var.eks_cpu_node_instance_type]
  }
  location_type = "availability-zone"
}

data "aws_ec2_instance_type_offerings" "availability_zones_gpu" {
  count = local.using_gpu ? 1 : 0
  filter {
    name   = "instance-type"
    values = [var.eks_gpu_node_instance_type]
  }
  location_type = "availability-zone"
}

resource "random_password" "db_password" {
  length           = 16
  special          = true
  override_special = "!#$%&*()-_=+[]{}<>:?"
}

module "kubeflow_utils" {
  source                             = "./kubeflow-utils"
  cluster_name                       = var.cluster_name
  db_allocated_storage               = var.db_allocated_storage
  db_backup_retention_period         = var.db_backup_retention_period
  db_class                           = var.db_class
  db_deletion_protection             = var.db_deletion_protection
  db_max_allocated_storage           = var.db_max_allocated_storage
  db_multi_az                        = var.db_multi_az
  db_mysql_engine_version            = var.db_mysql_engine_version
  db_name                            = var.db_name
  db_password                        = coalesce(var.db_password, try(random_password.db_password.result, null))
  db_publicly_accessible             = var.db_publicly_accessible
  db_secret_recovery_window_in_days  = var.db_secret_recovery_window_in_days
  db_storage_type                    = var.db_storage_type
  db_tags                            = var.db_tags
  db_username                        = var.db_username
  ecr_analytical_data_types          = var.ecr_analytical_data_types
  ecr_check_feature_distribution     = var.ecr_check_feature_distribution
  ecr_custom_predictor               = var.ecr_custom_predictor
  ecr_data_health_check              = var.ecr_data_health_check
  ecr_data_typing                    = var.ecr_data_typing
  ecr_evolutionary_algorithm         = var.ecr_evolutionary_algorithm
  ecr_feature_engineering            = var.ecr_feature_engineering
  ecr_feature_selector               = var.ecr_feature_selector
  ecr_image_classification_generator = var.ecr_image_classification_generator
  ecr_image_processor                = var.ecr_image_processor
  ecr_image_translation              = var.ecr_image_translation
  ecr_imputation                     = var.ecr_imputation
  ecr_interactive_visualizer         = var.ecr_interactive_visualizer
  ecr_model_evaluation               = var.ecr_model_evaluation
  ecr_model_generator_clustering     = var.ecr_model_generator_clustering
  ecr_model_generator_grow_net       = var.ecr_model_generator_grow_net
  ecr_model_generator_supervised     = var.ecr_model_generator_supervised
  ecr_model_registry                 = var.ecr_model_registry
  ecr_natural_language_processing    = var.ecr_natural_language_processing
  ecr_parallelizer                   = var.ecr_parallelizer
  ecr_sampling                       = var.ecr_sampling
  ecr_serializer                     = var.ecr_serializer
  ecr_slack_alerting                 = var.ecr_slack_alerting
  ecr_text_classification_generator  = var.ecr_text_classification_generator
  ecr_suffix                         = var.ecr_suffix
  environment                        = var.environment
  http_header_name                   = var.http_header_name
  load_balancer_scheme               = var.load_balancer_scheme
  s3_bucket_kubeflow_artifact_store  = var.s3_bucket_kubeflow_artifact_store
  s3_bucket_ml_ops_interim           = var.s3_bucket_ml_ops_interim
  s3_bucket_ml_ops_model_store       = var.s3_bucket_ml_ops_model_store
  s3_bucket_tag_name                 = var.s3_bucket_tag_name
  security_group_id                  = module.eks_blueprints.cluster_primary_security_group_id
  subnet_ids                         = module.vpc.private_subnets
  tags                               = var.tags
  use_rds                            = var.use_rds
  use_s3                             = var.use_s3
  vpc_id                             = module.vpc.vpc_id
}

module "kubeflow_components" {
  source                                       = "./kubeflow-components"
  addon_context                                = module.eks_blueprints_outputs.addon_context
  cluster_name                                 = var.cluster_name
  cognito_user_pool_client_allowed_oauth_flows = var.cognito_user_pool_client_allowed_oauth_flows
  cognito_user_pool_name                       = var.cognito_user_pool_name
  domain_name                                  = var.domain_name
  environment                                  = var.environment
  environment_sub_domain_name                  = var.environment_sub_domain_name
  http_header_name                             = var.http_header_name
  kf_helm_repo_path                            = local.kf_helm_repo_path
  kubeflow_db_address                          = module.kubeflow_utils.kubeflow_db_address
  load_balancer_scheme                         = var.load_balancer_scheme
  namespace_sub_domain_name                    = var.namespace_sub_domain_name
  notebook_cull_idle_time                      = var.notebook_cull_idle_time
  notebook_enable_culling                      = var.notebook_enable_culling
  notebook_idleness_check_period               = var.notebook_idleness_check_period
  pipeline_s3_credential_option                = var.pipeline_s3_credential_option
  rds_secret                                   = module.kubeflow_utils.rds_secret
  s3_bucket_kubeflow_artifact_store            = var.s3_bucket_kubeflow_artifact_store
  subnet_ids                                   = module.vpc.private_subnets
  tags                                         = var.tags
  top_level_domain_name                        = var.top_level_domain_name
  use_aws_telemetry                            = var.use_aws_telemetry
  use_cognito                                  = var.use_cognito
  use_rds                                      = var.use_rds
  use_s3                                       = var.use_s3
  vpc_id                                       = module.vpc.vpc_id
}

module "monitoring" {
  source                                            = "./monitoring"
  cluster_name                                      = var.cluster_name
  cluster_region                                    = var.cluster_region
  grafana_workspace_account_access_type             = var.grafana_workspace_account_access_type
  grafana_workspace_auth_providers                  = var.grafana_workspace_auth_providers
  grafana_workspace_data_sources                    = var.grafana_workspace_data_sources
  grafana_workspace_name                            = var.grafana_workspace_name
  grafana_workspace_notification_destinations       = var.grafana_workspace_notification_destinations
  grafana_workspace_permission_type                 = var.grafana_workspace_permission_type
  prometheus_workspace_name                         = var.prometheus_workspace_name
  cloudwatch_prometheus_log_group_retention_in_days = var.cloudwatch_prometheus_log_group_retention_in_days
  #depends_on = [module.kubeflow_components]
}

#######
# VPC #
#######

module "vpc" {
  source                        = "terraform-aws-modules/vpc/aws"
  version                       = "5.0.0"
  name                          = local.cluster_name
  cidr                          = local.vpc_cidr
  azs                           = local.azs
  public_subnets                = [for k, v in local.azs : cidrsubnet(local.vpc_cidr, 3, k)]
  private_subnets               = [for k, v in local.azs : cidrsubnet(local.vpc_cidr, 3, k + length(local.azs))]
  enable_nat_gateway            = true
  single_nat_gateway            = true
  enable_dns_hostnames          = true
  manage_default_network_acl    = true
  default_network_acl_tags      = { Name = "${local.cluster_name}-default" }
  manage_default_route_table    = true
  default_route_table_tags      = { Name = "${local.cluster_name}-default" }
  manage_default_security_group = true
  default_security_group_tags   = { Name = "${local.cluster_name}-default" }
  public_subnet_tags            = {
    "kubernetes.io/cluster/${local.cluster_name}" = "shared"
    "kubernetes.io/role/elb"                      = "1"
  }
  private_subnet_tags = {
    "kubernetes.io/cluster/${local.cluster_name}" = "shared"
    "kubernetes.io/role/internal-elb"             = "1"
  }
  tags = local.tags
}
