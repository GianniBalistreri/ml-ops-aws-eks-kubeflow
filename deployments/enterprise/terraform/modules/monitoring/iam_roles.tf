##########################
# AMP SigV4 Kiali Proxy: #
##########################

resource "aws_iam_role" "amp_sigv4_kiali_proxy" {
  name = "amp-sigv4-kiali-proxy"

  assume_role_policy = jsonencode({
    Version   = "2012-10-17",
    Statement = [
      {
        Action    = "sts:AssumeRole",
        Effect    = "Allow",
        Principal = {
          Service = "eks.amazonaws.com"
        }
      }
    ]
  })
}

data "aws_iam_policy_document" "amp_sigv4_kiali_proxy" {
  statement {
    actions   = [
      "aps:*",
      "cloudwatch:*",
      "grafana:*"
    ]
    resources = ["*"]
  }
}

resource "aws_iam_policy" "amp_sigv4_kiali_proxy" {
  name        = "amp-sigv4-kiali-proxy"
  description = "IAM policy for enabling Kiali to communicate wih AMP via Sigv4 proxy"
  policy      = data.aws_iam_policy_document.amp_sigv4_kiali_proxy.json
}

resource "aws_iam_role_policy_attachment" "amp_sigv4_kiali_proxy" {
  policy_arn = aws_iam_policy.amp_sigv4_kiali_proxy.arn
  role       = aws_iam_role.amp_sigv4_kiali_proxy.name
}

########
# AMP: #
########

resource "aws_iam_role" "prometheus" {
  name = "prometheus"

  assume_role_policy = jsonencode({
    Version   = "2012-10-17",
    Statement = [
      {
        Action    = "sts:AssumeRole",
        Effect    = "Allow",
        Principal = {
          Service = "eks.amazonaws.com"
        }
      }
    ]
  })
}

data "aws_iam_policy_document" "prometheus" {
  statement {
    actions   = [
      "aps:*",
      "cloudwatch:*"
      #"aps:RemoteWrite",
      #"aps:DescribeWorkspace",
      #"aps:QueryMetrics",
      #"aps:GetSeries",
      #"aps:GetLabels",
      #"aps:GetMetricMetadata"
    ]
    resources = ["*"]
  }
}

resource "aws_iam_policy" "prometheus" {
  name        = "prometheus"
  description = "IAM policy for enabling Prometheus to get metrics"
  policy      = data.aws_iam_policy_document.prometheus.json
}

resource "aws_iam_role_policy_attachment" "prometheus" {
  policy_arn = aws_iam_policy.prometheus.arn
  role       = aws_iam_role.prometheus.name
}

########
# AMG: #
########

resource "aws_iam_role" "grafana" {
  name = "grafana"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow",
        Principal = {
          Service = "grafana.amazonaws.com"
        }
      },
    ]
  })
}

data "aws_iam_policy_document" "grafana" {
  statement {
    actions   = [
      "aps:*",
      "cloudwatch:*",
      "grafana:*"
      #"aps:ListWorkspaces",
      #"aps:DescribeWorkspace",
      #"aps:QueryMetrics",
      #"aps:GetSeries",
      #"aps:GetLabels",
      #"aps:GetMetricMetadata"
    ]
    resources = ["*"]
  }
}

resource "aws_iam_policy" "grafana" {
  name        = "grafana"
  description = "IAM policy for enabling Grafana to connect with Prometheus"
  policy      = data.aws_iam_policy_document.grafana.json
}

resource "aws_iam_role_policy_attachment" "grafana" {
  policy_arn = aws_iam_policy.grafana.arn
  role       = aws_iam_role.grafana.name
}
