locals {
  cluster_name      = var.cluster_name
  region            = var.cluster_region
  eks_version       = var.eks_version
  vpc_cidr          = "10.0.0.0/16"
  using_gpu         = var.eks_gpu_node_instance_type != null
  available_azs_cpu = toset(data.aws_ec2_instance_type_offerings.availability_zones_cpu.locations)
  available_azs_gpu = toset(try(data.aws_ec2_instance_type_offerings.availability_zones_gpu[0].locations, []))
  available_azs     = local.using_gpu ? tolist(setintersection(local.available_azs_cpu, local.available_azs_gpu)) : tolist(local.available_azs_cpu)
  az_count          = min(length(local.available_azs), 3)
  azs               = slice(local.available_azs, 0, local.az_count)
  kf_helm_repo_path = var.kf_helm_repo_path
  tags              = {
    Platform        = "kubeflow-on-aws"
    KubeflowVersion = var.kubeflow_version
  }
  managed_node_group_cpu = {
    node_group_name = var.eks_cpu_nodegroup_name
    instance_types  = [var.eks_cpu_node_instance_type]
    min_size        = var.eks_cpu_min_size
    desired_size    = var.eks_cpu_desired_size
    max_size        = var.eks_cpu_max_size
    subnet_ids      = module.vpc.private_subnets
  }
  managed_node_group_gpu = local.using_gpu ? {
    node_group_name = var.eks_gpu_nodegroup_name
    instance_types  = [var.eks_gpu_node_instance_type]
    min_size        = var.eks_gpu_min_size
    desired_size    = var.eks_gpu_desired_size
    max_size        = var.eks_gpu_max_size
    ami_type        = var.eks_gpu_ami_type
    subnet_ids      = module.vpc.private_subnets
  } : null
  potential_managed_node_groups = {
    mg_cpu = local.managed_node_group_cpu,
    mg_gpu = local.managed_node_group_gpu
  }
  managed_node_groups = {for k, v in local.potential_managed_node_groups : k => v if v != null}
}

##################
# EKS Blueprints #
##################

module "eks_blueprints" {
  source              = "../../../../terraform-aws-eks-blueprints-4.32.1"
  cluster_name        = local.cluster_name
  cluster_version     = local.eks_version
  vpc_id              = module.vpc.vpc_id
  private_subnet_ids  = module.vpc.private_subnets
  managed_node_groups = local.managed_node_groups
  tags                = local.tags
}

module "ebs_csi_driver_irsa" {
  source                = "../../../../iaac/terraform/aws-infra/ebs-csi-driver-irsa"
  cluster_name          = local.cluster_name
  cluster_region        = local.region
  tags                  = local.tags
  eks_oidc_provider_arn = module.eks_blueprints.eks_oidc_provider_arn
  #depends_on            = [module.eks_blueprints]
}

module "eks_blueprints_kubernetes_addons" {
  source            = "aws-ia/eks-blueprints-addons/aws"
  version           = "~> 1.0"
  cluster_name      = local.cluster_name
  cluster_endpoint  = module.eks_blueprints.eks_cluster_endpoint
  cluster_version   = module.eks_blueprints.eks_cluster_version
  oidc_provider_arn = module.eks_blueprints.eks_oidc_provider_arn
  depends_on        = [module.ebs_csi_driver_irsa, module.eks_data_addons]
  eks_addons        = {
    aws-ebs-csi-driver = {
      most_recent              = true
      service_account_role_arn = module.ebs_csi_driver_irsa.iam_role_arn
    }
    coredns = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
  }
  enable_aws_load_balancer_controller = true
  enable_cert_manager                 = true
  cert_manager                        = {
    chart_version = "v1.10.0"
  }
  enable_aws_efs_csi_driver = true
  enable_aws_fsx_csi_driver = true
  aws_efs_csi_driver        = {
    namespace     = "kube-system"
    chart_version = "2.4.1"
  }
  aws_load_balancer_controller = {
    chart_version = "v1.4.8"
  }
  aws_fsx_csi_driver = {
    namespace     = "kube-system"
    chart_version = "1.5.1"
  }
  secrets_store_csi_driver = {
    namespace     = "kube-system"
    chart_version = "1.3.2"
    set           = [
      {
        name  = "syncSecret.enabled",
        value = "true"
      }
    ]
  }
  enable_secrets_store_csi_driver       = true
  secrets_store_csi_driver_provider_aws = {
    namespace = "kube-system"
    set       = [
      {
        name  = "secrets-store-csi-driver.install",
        value = "false"
      }
    ]
  }
  enable_secrets_store_csi_driver_provider_aws = true
  tags                                         = local.tags
}

module "eks_data_addons" {
  source                     = "aws-ia/eks-data-addons/aws"
  version                    = "~> 1.0"
  oidc_provider_arn          = module.eks_blueprints.eks_oidc_provider_arn
  enable_nvidia_gpu_operator = local.using_gpu
  #depends_on                 = [module.eks_blueprints]
}

#module "eks_blueprints_kubernetes_addons" {
#  source                               = "../../../../terraform-aws-eks-blueprints-4.32.1/modules/kubernetes-addons"
#  eks_cluster_id                       = module.eks_blueprints.eks_cluster_id
#  eks_cluster_endpoint                 = module.eks_blueprints.eks_cluster_endpoint
#  eks_oidc_provider                    = module.eks_blueprints.oidc_provider
#  eks_cluster_version                  = module.eks_blueprints.eks_cluster_version
# EKS Managed Add-ons
#  enable_amazon_eks_vpc_cni            = true
#  enable_amazon_eks_coredns            = true
#  enable_amazon_eks_kube_proxy         = true
#  enable_amazon_eks_aws_ebs_csi_driver = true
# EKS Blueprints Add-ons
#  enable_cert_manager                  = true
#  enable_aws_load_balancer_controller  = true

#  aws_efs_csi_driver_helm_config = {
#    namespace = "kube-system"
#    version   = "2.4.1"
#  }

#  enable_aws_efs_csi_driver      = true
#  aws_fsx_csi_driver_helm_config = {
#    namespace = "kube-system"
#    version   = "1.5.1"
#  }

#  enable_aws_fsx_csi_driver   = true
#  enable_nvidia_device_plugin = local.using_gpu

#  enable_secrets_store_csi_driver_provider_aws = true
#  tags                                         = local.tags
#}

module "eks_blueprints_outputs" {
  source               = "../../../../iaac/terraform/utils/blueprints-extended-outputs"
  eks_cluster_id       = module.eks_blueprints.eks_cluster_id
  eks_cluster_endpoint = module.eks_blueprints.eks_cluster_endpoint
  eks_oidc_provider    = module.eks_blueprints.oidc_provider
  eks_cluster_version  = module.eks_blueprints.eks_cluster_version
  tags                 = local.tags
}
