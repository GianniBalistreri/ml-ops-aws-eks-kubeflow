provider "aws" {
  alias  = "aws"
  region = var.cluster_region
}

provider "aws" {
  alias  = "virginia"
  region = "us-east-1"
}

resource "kubernetes_namespace" "kubeflow" {
  metadata {
    labels = {
      control-plane   = "kubeflow"
      istio-injection = "enabled"
    }
    name = "kubeflow"
  }
}

locals {
  katib_chart_vanilla = "${var.kf_helm_repo_path}/charts/apps/katib/vanilla"
  katib_chart_rds     = "${var.kf_helm_repo_path}/charts/apps/katib/katib-external-db-with-kubeflow"

  kfp_chart_vanilla           = "${var.kf_helm_repo_path}/charts/apps/kubeflow-pipelines/vanilla"
  kfp_chart_rds_only          = "${var.kf_helm_repo_path}/charts/apps/kubeflow-pipelines/rds-only"
  kfp_chart_s3_only           = "${var.kf_helm_repo_path}/charts/apps/kubeflow-pipelines/s3-only"
  kfp_chart_rds_and_s3        = "${var.kf_helm_repo_path}/charts/apps/kubeflow-pipelines/rds-s3"
  kfp_chart_s3_only_static    = "${var.kf_helm_repo_path}/charts/apps/kubeflow-pipelines/s3-only-static"
  kfp_chart_rds_and_s3_static = "${var.kf_helm_repo_path}/charts/apps/kubeflow-pipelines/rds-s3-static"

  secrets_manager_chart_rds    = "${var.kf_helm_repo_path}/charts/common/aws-secrets-manager/rds-only"
  secrets_manager_chart_s3     = "${var.kf_helm_repo_path}/charts/common/aws-secrets-manager/s3-only"
  secrets_manager_chart_rds_s3 = "${var.kf_helm_repo_path}/charts/common/aws-secrets-manager/rds-s3"

  use_static = "static" == var.pipeline_s3_credential_option
  use_irsa   = "irsa" == var.pipeline_s3_credential_option

  kfp_chart_map = {
    (local.kfp_chart_vanilla)           = !var.use_rds && !var.use_s3,
    (local.kfp_chart_rds_only)          = var.use_rds && !var.use_s3,
    (local.kfp_chart_s3_only)           = !var.use_rds && var.use_s3 && local.use_irsa,
    (local.kfp_chart_rds_and_s3)        = var.use_rds && var.use_s3 && local.use_irsa,
    (local.kfp_chart_s3_only_static)    = !var.use_rds && var.use_s3 && local.use_static,
    (local.kfp_chart_rds_and_s3_static) = var.use_rds && var.use_s3 && local.use_static
  }

  secrets_manager_chart_map = {
    (local.secrets_manager_chart_rds)    = var.use_rds && var.use_s3 && local.use_irsa,
    (local.secrets_manager_chart_s3)     = !var.use_rds && var.use_s3 && local.use_static,
    (local.secrets_manager_chart_rds_s3) = var.use_rds && var.use_s3 && local.use_static
  }

  katib_chart                      = var.use_rds ? local.katib_chart_rds : local.katib_chart_vanilla
  kfp_chart                        = [for k, v in local.kfp_chart_map : k if v == true][0]
  secrets_managers_possible_charts = [for k, v in local.secrets_manager_chart_map : k if v == true]
  secrets_manager_chart            = length(local.secrets_managers_possible_charts) > 0 ? local.secrets_managers_possible_charts[0] : ""
}

data "aws_iam_role" "pipeline_irsa_iam_role" {
  count      = local.use_static ? 0 : 1
  name       = try(module.kubeflow_pipeline_irsa[0].irsa_iam_role_name, null)
  depends_on = [module.kubeflow_pipeline_irsa]
}

data "aws_iam_role" "user_namespace_irsa_iam_role" {
  count      = local.use_static ? 0 : 1
  name       = try(module.user_namespace_irsa[0].irsa_iam_role_name, null)
  depends_on = [module.user_namespace_irsa]
}

module "kubeflow_secrets_manager_irsa" {
  count                             = local.use_static || var.use_rds ? 1 : 0
  source                            = "../../../../../terraform-aws-eks-blueprints-4.32.1/modules/irsa"
  kubernetes_namespace              = kubernetes_namespace.kubeflow.metadata[0].name
  create_kubernetes_namespace       = false
  create_kubernetes_service_account = true
  kubernetes_service_account        = "kubeflow-secrets-manager-sa"
  irsa_iam_role_name                = format("%s-%s-%s-%s", "kf-secrets-manager", "irsa", var.addon_context.eks_cluster_id, var.addon_context.aws_region_name)
  irsa_iam_policies                 = [
    "arn:aws:iam::aws:policy/AmazonSSMReadOnlyAccess", "arn:aws:iam::aws:policy/SecretsManagerReadWrite"
  ]
  irsa_iam_role_path            = var.addon_context.irsa_iam_role_path
  irsa_iam_permissions_boundary = var.addon_context.irsa_iam_permissions_boundary
  eks_cluster_id                = var.addon_context.eks_cluster_id
  eks_oidc_provider_arn         = var.addon_context.eks_oidc_provider_arn
}

module "kubeflow_pipeline_irsa" {
  count                             = local.use_static ? 0 : 1
  source                            = "../../../../../terraform-aws-eks-blueprints-4.32.1/modules/irsa"
  kubernetes_namespace              = kubernetes_namespace.kubeflow.metadata[0].name
  create_kubernetes_namespace       = false
  create_kubernetes_service_account = false
  kubernetes_service_account        = "ml-pipeline"
  irsa_iam_role_name                = format("%s-%s-%.22s-%.16s", "ml-pipeline", "irsa", var.addon_context.eks_cluster_id, var.addon_context.aws_region_name)
  irsa_iam_policies                 = ["arn:aws:iam::aws:policy/AmazonS3FullAccess"]
  irsa_iam_role_path                = var.addon_context.irsa_iam_role_path
  irsa_iam_permissions_boundary     = var.addon_context.irsa_iam_permissions_boundary
  eks_cluster_id                    = var.addon_context.eks_cluster_id
  eks_oidc_provider_arn             = var.addon_context.eks_oidc_provider_arn
}

module "user_namespace_irsa" {
  count                             = local.use_static ? 0 : 1
  source                            = "../../../../../terraform-aws-eks-blueprints-4.32.1/modules/irsa"
  kubernetes_namespace              = "kubeflow-user-example-com"
  create_kubernetes_namespace       = false
  create_kubernetes_service_account = false
  kubernetes_service_account        = "default-editor"
  irsa_iam_role_name                = format("%s-%s-%.22s-%.16s", "user-namespace", "irsa", var.addon_context.eks_cluster_id, var.addon_context.aws_region_name)
  irsa_iam_policies                 = ["arn:aws:iam::aws:policy/AmazonS3FullAccess"]
  irsa_iam_role_path                = var.addon_context.irsa_iam_role_path
  irsa_iam_permissions_boundary     = var.addon_context.irsa_iam_permissions_boundary
  eks_cluster_id                    = var.addon_context.eks_cluster_id
  eks_oidc_provider_arn             = var.addon_context.eks_oidc_provider_arn
}

module "filter_secrets_manager_set_values" {
  source     = "../../../../../iaac/terraform/utils/set-values-filter"
  set_values = {
    "rds.secretName" = try(var.rds_secret, null)
  }
  depends_on = [kubernetes_namespace.kubeflow]
}

module "secrets_manager" {
  count       = var.use_rds || (var.use_s3 && local.use_static) ? 1 : 0
  source      = "../../../../../iaac/terraform/common/aws-secrets-manager"
  helm_config = {
    chart = local.secrets_manager_chart
    set   = module.filter_secrets_manager_set_values.set_values
  }
  addon_context = var.addon_context
  depends_on    = [kubernetes_namespace.kubeflow]
}

module "subdomain" {
  source                      = "../subdomain"
  top_level_domain_name       = var.top_level_domain_name
  domain_name                 = var.domain_name
  environment_sub_domain_name = var.environment_sub_domain_name
  namespace_sub_domain_name   = var.namespace_sub_domain_name
}

module "kubeflow_issuer" {
  source      = "../../../../../iaac/terraform/common/kubeflow-issuer"
  helm_config = {
    chart = "${var.kf_helm_repo_path}/charts/common/kubeflow-issuer"
  }
  addon_context = var.addon_context
  depends_on    = [kubernetes_namespace.kubeflow]
}

module "kubeflow_istio" {
  source      = "../../../../../iaac/terraform/common/istio"
  helm_config = {
    chart = "${var.kf_helm_repo_path}/charts/common/istio"
  }
  addon_context = var.addon_context
  depends_on    = [module.kubeflow_issuer]
}

module "ingress_cognito" {
  count                                        = var.use_cognito ? 1 : 0
  source                                       = "../cognito"
  cognito_user_pool_name                       = var.cognito_user_pool_name
  cognito_user_pool_client_allowed_oauth_flows = var.cognito_user_pool_client_allowed_oauth_flows
  top_level_domain_name                        = var.top_level_domain_name
  domain_name                                  = var.domain_name
  environment_sub_domain_name                  = var.environment_sub_domain_name
  namespace_sub_domain_name                    = var.namespace_sub_domain_name
  callback_logout_sub_domain_name              = var.callback_logout_sub_domain_name
  load_balancer_scheme                         = var.load_balancer_scheme
  cluster_name                                 = var.cluster_name
  tags                                         = var.tags
  addon_context                                = var.addon_context
  providers                                    = {
    aws          = aws.aws
    aws.virginia = aws.virginia
  }
  depends_on = [module.subdomain, module.kubeflow_istio]
}

module "kubeflow_aws_authservice" {
  count       = var.use_cognito ? 1 : 0
  source      = "../../../../../iaac/terraform/common/aws-authservice"
  helm_config = {
    chart = "${var.kf_helm_repo_path}/charts/common/aws-authservice"
    set   = [
      {
        name  = "LOGOUT_URL"
        value = module.ingress_cognito[0].logout_url
      }
    ]
  }
  addon_context = var.addon_context
  depends_on    = [module.ingress_cognito]
}

module "load_balancer" {
  count                       = var.use_cognito ? 0 : 1
  source                      = "../load-balancer"
  cluster_name                = var.cluster_name
  http_header_name            = var.http_header_name
  load_balancer_scheme        = var.load_balancer_scheme
  top_level_domain_name       = var.top_level_domain_name
  domain_name                 = var.domain_name
  environment_sub_domain_name = var.environment_sub_domain_name
  namespace_sub_domain_name   = var.namespace_sub_domain_name
  depends_on                  = [module.subdomain, module.kubeflow_istio]
}

module "kubeflow_dex" {
  count       = var.use_cognito ? 0 : 1
  source      = "../../../../../iaac/terraform/common/dex"
  helm_config = {
    chart = "${var.kf_helm_repo_path}/charts/common/dex"
  }
  addon_context = var.addon_context
  depends_on    = [module.load_balancer]
}

module "kubeflow_oidc_authservice" {
  count       = var.use_cognito ? 0 : 1
  source      = "../../../../../iaac/terraform/common/oidc-authservice"
  helm_config = {
    chart = "${var.kf_helm_repo_path}/charts/common/oidc-authservice"
  }
  addon_context = var.addon_context
  depends_on    = [module.kubeflow_dex]
}

module "kubeflow_knative_serving" {
  source      = "../../../../../iaac/terraform/common/knative-serving"
  helm_config = {
    chart = "${var.kf_helm_repo_path}/charts/common/knative-serving"
  }
  addon_context = var.addon_context
  depends_on    = [module.kubeflow_oidc_authservice, module.ingress_cognito]
}

module "kubeflow_cluster_local_gateway" {
  source      = "../../../../../iaac/terraform/common/cluster-local-gateway"
  helm_config = {
    chart = "${var.kf_helm_repo_path}/charts/common/cluster-local-gateway"
  }
  addon_context = var.addon_context
  depends_on    = [module.kubeflow_knative_serving]
}

module "kubeflow_knative_eventing" {
  source      = "../../../../../iaac/terraform/common/knative-eventing"
  helm_config = {
    chart = "${var.kf_helm_repo_path}/charts/common/knative-eventing"
  }
  addon_context = var.addon_context
  depends_on    = [module.kubeflow_cluster_local_gateway]
}

module "kubeflow_roles" {
  source      = "../../../../../iaac/terraform/common/kubeflow-roles"
  helm_config = {
    chart = "${var.kf_helm_repo_path}/charts/common/kubeflow-roles"
  }
  addon_context = var.addon_context
  depends_on    = [module.kubeflow_knative_eventing]
}

module "kubeflow_istio_resources" {
  source      = "../../../../../iaac/terraform/common/kubeflow-istio-resources"
  helm_config = {
    chart = "${var.kf_helm_repo_path}/charts/common/kubeflow-istio-resources"
  }
  addon_context = var.addon_context
  depends_on    = [module.kubeflow_roles]
}

module "filter_kfp_set_values" {
  source     = "../../../../../iaac/terraform/utils/set-values-filter"
  set_values = {
    "rds.dbHost"            = try(var.kubeflow_db_address, null),
    "s3.bucketName"         = var.s3_bucket_kubeflow_artifact_store
    "s3.minioServiceRegion" = coalesce(null, var.addon_context.aws_region_name)
    "rds.mlmdDb"            = "metadb",
    "s3.minioServiceHost"   = "s3.amazonaws.com"
    "s3.roleArn"            = try(data.aws_iam_role.pipeline_irsa_iam_role[0].arn, null)
  }
}

module "kubeflow_pipelines" {
  source      = "../../../../../iaac/terraform/apps/kubeflow-pipelines"
  helm_config = {
    chart = local.kfp_chart
    set   = module.filter_kfp_set_values.set_values
  }
  addon_context = var.addon_context
  depends_on    = [module.kubeflow_istio_resources, module.secrets_manager, module.kubeflow_pipeline_irsa]
}

module "kubeflow_kserve" {
  source      = "../../../../../iaac/terraform/common/kserve"
  helm_config = {
    chart = "${var.kf_helm_repo_path}/charts/common/kserve"
  }
  addon_context = var.addon_context
  depends_on    = [module.kubeflow_pipelines]
}

module "kubeflow_models_web_app" {
  source      = "../../../../../iaac/terraform/apps/models-web-app"
  helm_config = {
    chart = "${var.kf_helm_repo_path}/charts/apps/models-web-app"
  }
  addon_context = var.addon_context
  depends_on    = [module.kubeflow_kserve]
}

module "kubeflow_katib" {
  source      = "../../../../../iaac/terraform/apps/katib"
  helm_config = {
    chart = "${var.kf_helm_repo_path}/charts/apps/katib/vanilla"
  }
  addon_context = var.addon_context
  depends_on    = [module.kubeflow_models_web_app]
}

module "kubeflow_central_dashboard" {
  source      = "../../../../../iaac/terraform/apps/central-dashboard"
  helm_config = {
    chart = "${var.kf_helm_repo_path}/charts/apps/central-dashboard"
  }
  addon_context = var.addon_context
  depends_on    = [module.kubeflow_katib]
}

module "kubeflow_admission_webhook" {
  source      = "../../../../../iaac/terraform/apps/admission-webhook"
  helm_config = {
    chart = "${var.kf_helm_repo_path}/charts/apps/admission-webhook"
  }
  addon_context = var.addon_context
  depends_on    = [module.kubeflow_central_dashboard]
}

module "kubeflow_notebook_controller" {
  source      = "../../../../../iaac/terraform/apps/notebook-controller"
  helm_config = {
    chart = "${var.kf_helm_repo_path}/charts/apps/notebook-controller"
    set   = [
      {
        name  = "cullingPolicy.cullIdleTime",
        value = var.notebook_cull_idle_time
      },
      {
        name  = "cullingPolicy.enableCulling",
        value = var.notebook_enable_culling
      },
      {
        name  = "cullingPolicy.idlenessCheckPeriod",
        value = var.notebook_idleness_check_period
      }
    ]
  }
  addon_context = var.addon_context
  depends_on    = [module.kubeflow_admission_webhook]
}

module "kubeflow_jupyter_web_app" {
  source      = "../../../../../iaac/terraform/apps/jupyter-web-app"
  helm_config = {
    chart = "${var.kf_helm_repo_path}/charts/apps/jupyter-web-app"
  }
  addon_context = var.addon_context
  depends_on    = [module.kubeflow_notebook_controller]
}

module "kubeflow_profiles_and_kfam" {
  source      = "../../../../../iaac/terraform/apps/profiles-and-kfam"
  helm_config = {
    chart = "${var.kf_helm_repo_path}/charts/apps/profiles-and-kfam"
  }
  addon_context = var.addon_context
  depends_on    = [module.kubeflow_notebook_controller]
}

module "kubeflow_volumes_web_app" {
  source      = "../../../../../iaac/terraform/apps/volumes-web-app"
  helm_config = {
    chart = "${var.kf_helm_repo_path}/charts/apps/volumes-web-app"
  }
  addon_context = var.addon_context
  depends_on    = [module.kubeflow_profiles_and_kfam]
}

module "kubeflow_tensorboards_web_app" {
  source      = "../../../../../iaac/terraform/apps/tensorboards-web-app"
  helm_config = {
    chart = "${var.kf_helm_repo_path}/charts/apps/tensorboards-web-app"
  }
  addon_context = var.addon_context
  depends_on    = [module.kubeflow_volumes_web_app]
}

module "kubeflow_tensorboard_controller" {
  source      = "../../../../../iaac/terraform/apps/tensorboard-controller"
  helm_config = {
    chart = "${var.kf_helm_repo_path}/charts/apps/tensorboard-controller"
  }
  addon_context = var.addon_context
  depends_on    = [module.kubeflow_tensorboards_web_app]
}

module "kubeflow_training_operator" {
  source      = "../../../../../iaac/terraform/apps/training-operator"
  helm_config = {
    chart = "${var.kf_helm_repo_path}/charts/apps/training-operator"
  }
  addon_context = var.addon_context
  depends_on    = [module.kubeflow_tensorboard_controller]
}

module "kubeflow_aws_telemetry" {
  count       = var.use_aws_telemetry ? 1 : 0
  source      = "../../../../../iaac/terraform/common/aws-telemetry"
  helm_config = {
    chart = "${var.kf_helm_repo_path}/charts/common/aws-telemetry"
  }
  addon_context = var.addon_context
  depends_on    = [module.kubeflow_training_operator]
}

module "filter_kubeflow_user_namespace_set_values" {
  source     = "../../../../../iaac/terraform/utils/set-values-filter"
  set_values = {
    "awsIamForServiceAccount.awsIamRole" = try(data.aws_iam_role.user_namespace_irsa_iam_role[0].arn, null)
  }
}

module "kubeflow_user_namespace" {
  source      = "../../../../../iaac/terraform/common/user-namespace"
  helm_config = {
    chart = "${var.kf_helm_repo_path}/charts/common/user-namespace"
    set   = module.filter_kubeflow_user_namespace_set_values.set_values
  }
  addon_context = var.addon_context
  depends_on    = [module.kubeflow_aws_telemetry, module.user_namespace_irsa]
}

#module "ack_sagemaker" {
#  source        = "../../../../../iaac/terraform/common/ack-sagemaker-controller"
#  addon_context = var.addon_context
#}
