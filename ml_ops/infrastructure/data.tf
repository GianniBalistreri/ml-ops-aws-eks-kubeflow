# Get Account Identity:
data "aws_caller_identity" "current" {}
# Fetch AZs in the current region
data "aws_availability_zones" "available" {
}

data "aws_eks_cluster" "cluster" {
  name = module.eks.cluster_id
}

data "aws_eks_cluster_auth" "cluster" {
  name = module.eks.cluster_id
}