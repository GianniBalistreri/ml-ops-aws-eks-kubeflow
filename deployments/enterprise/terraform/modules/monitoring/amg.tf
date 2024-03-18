#resource "aws_grafana_workspace" "kubeflow" {
#  name                      = var.grafana_workspace_name
#  account_access_type       = var.grafana_workspace_account_access_type
#  authentication_providers  = var.grafana_workspace_auth_providers
#  permission_type           = var.grafana_workspace_permission_type
#  role_arn                  = aws_iam_role.grafana.arn
#  data_sources              = var.grafana_workspace_data_sources
#  notification_destinations = var.grafana_workspace_notification_destinations

#  depends_on = [aws_iam_role_policy_attachment.grafana]
#}
