###########################
# Kubeflow User IAM Role: #
###########################

resource "aws_iam_role" "kubeflow_user" {
  name = "kubeflow-user-role"

  assume_role_policy = jsonencode({
    Version   = "2012-10-17",
    Statement = [
      {
        Action    = "sts:AssumeRole",
        Effect    = "Allow",
        Principal = {
          Service = "eks.amazonaws.com"
        }
      },
      {
        "Effect": "Allow",
        "Principal": {
          "Federated": module.kubeflow_pipeline_irsa[0].eks_oidc_provider_arn
        },
        "Action": "sts:AssumeRoleWithWebIdentity"
      }
    ]
  })
}

#######
# S3: #
#######

resource "aws_iam_policy" "s3" {
  name        = "s3-policy"
  description = "Policy for S3 permissions"

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action   = "s3:*",
        Effect   = "Allow",
        Resource = "arn:aws:s3:::*"
      },
      {
        Action   = "s3:*",
        Effect   = "Deny",
        Resource = "arn:aws:s3:::shopware-data-lake*/*"
      },
      {
        Action   = "s3:*",
        Effect   = "Deny",
        Resource = "arn:aws:s3:::shopware-data-utils*/*"
      },
      {
        Action   = "s3:*",
        Effect   = "Deny",
        Resource = "arn:aws:s3:::shopware-raw-tracking-data*/*"
      },
      {
        Action   = "s3:*",
        Effect   = "Deny",
        Resource = "arn:aws:s3:::shopware-reporting*/*"
      },
      {
        Action   = "s3:*",
        Effect   = "Deny",
        Resource = "arn:aws:s3:::shopware-ml-ops-tfstate*/*"
      },
      {
        Action   = "s3:*",
        Effect   = "Deny",
        Resource = "arn:aws:s3:::shopware-tracking-gateway-tfstate*/*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "s3" {
  policy_arn = aws_iam_policy.s3.arn
  role       = aws_iam_role.kubeflow_user.name
}

########
# EKS: #
########

data "aws_iam_policy_document" "eks" {
  statement {
    actions   = ["eks:Describe*", "eks:List*"]
    resources = ["*"]
  }
}

resource "aws_iam_policy" "eks" {
  name        = "eks-policy"
  description = "Policy for EKS permissions"
  policy      = data.aws_iam_policy_document.eks.json
}

resource "aws_iam_role_policy_attachment" "eks" {
  policy_arn = aws_iam_policy.eks.arn
  role       = aws_iam_role.kubeflow_user.name
}

########
# EC2: #
########

data "aws_iam_policy_document" "ec2" {
  statement {
    actions   = ["ec2:Describe*", "ec2:List*"]
    resources = ["*"]
  }
}

resource "aws_iam_policy" "ec2" {
  name        = "ec2-policy"
  description = "Policy for EC2 permissions"
  policy      = data.aws_iam_policy_document.ec2.json
}

resource "aws_iam_role_policy_attachment" "ec2" {
  policy_arn = aws_iam_policy.ec2.arn
  role       = aws_iam_role.kubeflow_user.name
}

########
# ECR: #
########

data "aws_iam_policy_document" "ecr" {
  statement {
    actions = [
      "ecr:GetDownloadUrlForLayer",
      "ecr:GetRepositoryPolicy",
      "ecr:DescribeRepositories",
      "ecr:ListImages",
      "ecr:BatchCheckLayerAvailability",
      "ecr:GetLifecyclePolicy",
      "ecr:GetLifecyclePolicyPreview",
      "ecr:GetRepositoryPolicy",
      "ecr:ListTagsForResource",
      "ecr:DescribeImageScanFindings",
      "ecr:InitiateLayerUpload",
      "ecr:UploadLayerPart",
      "ecr:CompleteLayerUpload",
      "ecr:PutImage"
    ]
    resources = ["*"]
  }
}

resource "aws_iam_policy" "ecr" {
  name        = "ecr-policy"
  description = "Policy for ECR permissions"
  policy      = data.aws_iam_policy_document.ecr.json
}

resource "aws_iam_role_policy_attachment" "ecr" {
  policy_arn = aws_iam_policy.ecr.arn
  role       = aws_iam_role.kubeflow_user.name
}

#########
# Glue: #
#########

data "aws_iam_policy_document" "glue" {
  statement {
    actions   = ["glue:Get*", "glue:List*"]
    resources = ["*"]
  }
}

resource "aws_iam_policy" "glue" {
  name        = "glue-policy"
  description = "Policy for AWS Glue permissions"
  policy      = data.aws_iam_policy_document.glue.json
}

resource "aws_iam_role_policy_attachment" "glue" {
  policy_arn = aws_iam_policy.glue.arn
  role       = aws_iam_role.kubeflow_user.name
}

data "aws_iam_role" "kubeflow_managed_ondemand_cpu" {
  name = "kubeflow-managed-ondemand-cpu"

  depends_on = [kubernetes_namespace.kubeflow]
}

resource "aws_iam_policy" "istio_ingress" {
  name        = "istio-ingress"
  description = "Permission to attach nodes to ALB"

  policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "elasticloadbalancing:DescribeLoadBalancers",
        "ec2:DescribeAvailabilityZones",
        "acm:GetCertificate",
        "acm:ListCertificates"
      ],
      "Resource": "*"
    }
  ]
}
EOF
}

resource "aws_iam_policy_attachment" "kubeflow_managed_ondemand_cpu" {
  name       = "istio-ingress-attachment"
  policy_arn = aws_iam_policy.istio_ingress.arn
  roles      = [data.aws_iam_role.kubeflow_managed_ondemand_cpu.name]
}

resource "aws_iam_policy" "alb_controller_subnet_access" {
  name        = "alb-controller-subnet-access"
  description = "Additional policies to enable ALB controller to access EKS subnets"

  policy = jsonencode({
    Version   = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = [
          "iam:CreateServiceLinkedRole"
        ],
        Resource  = "*",
        Condition = {
          StringEquals = {
            "iam:AWSServiceName" = "elasticloadbalancing.amazonaws.com"
          }
        }
      },
      {
        Effect = "Allow",
        Action = [
          "ec2:DescribeAccountAttributes",
          "ec2:DescribeAddresses",
          "ec2:DescribeAvailabilityZones",
          "ec2:DescribeInternetGateways",
          "ec2:DescribeVpcs",
          "ec2:DescribeVpcPeeringConnections",
          "ec2:DescribeSubnets",
          "ec2:DescribeSecurityGroups",
          "ec2:DescribeInstances",
          "ec2:DescribeNetworkInterfaces",
          "ec2:DescribeTags",
          "ec2:GetCoipPoolUsage",
          "ec2:DescribeCoipPools",
          "elasticloadbalancing:DescribeLoadBalancers",
          "elasticloadbalancing:DescribeLoadBalancerAttributes",
          "elasticloadbalancing:DescribeListeners",
          "elasticloadbalancing:DescribeListenerCertificates",
          "elasticloadbalancing:DescribeSSLPolicies",
          "elasticloadbalancing:DescribeRules",
          "elasticloadbalancing:DescribeTargetGroups",
          "elasticloadbalancing:DescribeTargetGroupAttributes",
          "elasticloadbalancing:DescribeTargetHealth",
          "elasticloadbalancing:DescribeTags"
        ],
        Resource = "*"
      },
      {
        Effect = "Allow",
        Action = [
          "cognito-idp:DescribeUserPoolClient",
          "acm:GetCertificate",
          "acm:ListCertificates",
          "acm:DescribeCertificate",
          "iam:ListServerCertificates",
          "iam:GetServerCertificate",
          "waf-regional:GetWebACL",
          "waf-regional:GetWebACLForResource",
          "waf-regional:AssociateWebACL",
          "waf-regional:DisassociateWebACL",
          "wafv2:GetWebACL",
          "wafv2:GetWebACLForResource",
          "wafv2:AssociateWebACL",
          "wafv2:DisassociateWebACL",
          "shield:GetSubscriptionState",
          "shield:DescribeProtection",
          "shield:CreateProtection",
          "shield:DeleteProtection"
        ],
        Resource = "*"
      },
      {
        Effect = "Allow",
        Action = [
          "ec2:AuthorizeSecurityGroupIngress",
          "ec2:RevokeSecurityGroupIngress"
        ],
        Resource = "*"
      },
      {
        Effect = "Allow",
        Action = [
          "ec2:CreateSecurityGroup"
        ],
        Resource = "*"
      },
      {
        Effect = "Allow",
        Action = [
          "ec2:CreateTags"
        ],
        Resource  = "arn:aws:ec2:*:*:security-group/*",
        Condition = {
          StringEquals = {
            "ec2:CreateAction" = "CreateSecurityGroup"
          },
          Null = {
            "aws:RequestTag/elbv2.k8s.aws/cluster" = "false"
          }
        }
      },
      {
        Effect = "Allow",
        Action = [
          "ec2:CreateTags",
          "ec2:DeleteTags"
        ],
        Resource  = "arn:aws:ec2:*:*:security-group/*",
        Condition = {
          Null = {
            "aws:RequestTag/elbv2.k8s.aws/cluster"  = "true",
            "aws:ResourceTag/elbv2.k8s.aws/cluster" = "false"
          }
        }
      },
      {
        Effect = "Allow",
        Action = [
          "ec2:AuthorizeSecurityGroupIngress",
          "ec2:RevokeSecurityGroupIngress",
          "ec2:DeleteSecurityGroup"
        ],
        Resource  = "*",
        Condition = {
          Null = {
            "aws:ResourceTag/elbv2.k8s.aws/cluster" = "false"
          }
        }
      },
      {
        Effect = "Allow",
        Action = [
          "elasticloadbalancing:CreateLoadBalancer",
          "elasticloadbalancing:CreateTargetGroup"
        ],
        Resource  = "*",
        Condition = {
          Null = {
            "aws:RequestTag/elbv2.k8s.aws/cluster" = "false"
          }
        }
      },
      {
        Effect = "Allow",
        Action = [
          "elasticloadbalancing:CreateListener",
          "elasticloadbalancing:DeleteListener",
          "elasticloadbalancing:CreateRule",
          "elasticloadbalancing:DeleteRule",
          "elasticloadbalancing:AddTags",
          "elasticloadbalancing:RemoveTags"
        ],
        Resource = "*"
      },
      {
        Effect = "Allow",
        Action = [
          "elasticloadbalancing:AddTags",
          "elasticloadbalancing:RemoveTags"
        ],
        Resource = [
          "arn:aws:elasticloadbalancing:*:*:targetgroup/*/*",
          "arn:aws:elasticloadbalancing:*:*:loadbalancer/net/*/*",
          "arn:aws:elasticloadbalancing:*:*:loadbalancer/app/*/*"
        ],
        Condition = {
          Null = {
            "aws:RequestTag/elbv2.k8s.aws/cluster"  = "true",
            "aws:ResourceTag/elbv2.k8s.aws/cluster" = "false"
          }
        }
      },
      {
        Effect = "Allow",
        Action = [
          "elasticloadbalancing:AddTags",
          "elasticloadbalancing:RemoveTags"
        ],
        Resource = [
          "arn:aws:elasticloadbalancing:*:*:listener/net/*/*/*",
          "arn:aws:elasticloadbalancing:*:*:listener/app/*/*/*",
          "arn:aws:elasticloadbalancing:*:*:listener-rule/net/*/*/*",
          "arn:aws:elasticloadbalancing:*:*:listener-rule/app/*/*/*"
        ]
      },
      {
        Effect = "Allow",
        Action = [
          "elasticloadbalancing:ModifyLoadBalancerAttributes",
          "elasticloadbalancing:SetIpAddressType",
          "elasticloadbalancing:SetSecurityGroups",
          "elasticloadbalancing:SetSubnets",
          "elasticloadbalancing:DeleteLoadBalancer",
          "elasticloadbalancing:ModifyTargetGroup",
          "elasticloadbalancing:ModifyTargetGroupAttributes",
          "elasticloadbalancing:DeleteTargetGroup"
        ],
        Resource  = "*",
        Condition = {
          Null = {
            "aws:ResourceTag/elbv2.k8s.aws/cluster" = "false"
          }
        }
      },
      {
        Effect = "Allow",
        Action = [
          "elasticloadbalancing:RegisterTargets",
          "elasticloadbalancing:DeregisterTargets"
        ],
        Resource = "arn:aws:elasticloadbalancing:*:*:targetgroup/*/*"
      },
      {
        Effect = "Allow",
        Action = [
          "elasticloadbalancing:SetWebAcl",
          "elasticloadbalancing:ModifyListener",
          "elasticloadbalancing:AddListenerCertificates",
          "elasticloadbalancing:RemoveListenerCertificates",
          "elasticloadbalancing:ModifyRule",
        ],
        Resource = "*"
      }
    ]
  })
}
