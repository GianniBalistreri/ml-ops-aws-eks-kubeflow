resource "aws_cloudwatch_log_group" "prometheus" {
  name              = "/aws/prometheus/kubeflow"
  retention_in_days = var.cloudwatch_prometheus_log_group_retention_in_days
}
