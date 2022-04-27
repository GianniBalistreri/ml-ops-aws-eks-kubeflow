# Get Account Identity:
data "aws_caller_identity" "current" {}

# Fetch AZs in the current region:
data "aws_availability_zones" "available" {}

# Get AWS EKS cluster id:
data "aws_eks_cluster" "cluster" {
  name = module.eks.cluster_id
}

data "aws_eks_cluster_auth" "cluster" {
  name = module.eks.cluster_id
}

# Github action:
data "aws_iam_policy_document" "github_allow" {
  statement {
    effect  = "Allow"
    actions = [
      "sts:AssumeRole",
      "sts:AssumeRoleWithWebIdentity"
      ]
    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.githubOidc.arn]
    }
    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:GianniBalistreri/ml-ops-aws-eks-kubeflow:*"]
    }
  }
}
