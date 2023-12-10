resource "aws_prometheus_workspace" "kubeflow" {
  alias = var.prometheus_workspace_name

  logging_configuration {
    log_group_arn = "${aws_cloudwatch_log_group.prometheus.arn}:*"
  }

  depends_on = [null_resource.create_amp_service_account]
}

resource "null_resource" "prometheus" {
  triggers = {
    always_run = "${timestamp()}"
  }

  provisioner "local-exec" {
    command = <<EOT
      export AMP_WORKSPACE_REGION=$(echo ${aws_prometheus_workspace.kubeflow.arn} | cut -d':' -f4)
      export AMP_WORKSPACE_ID=$(echo ${aws_prometheus_workspace.kubeflow.arn} | cut -d':' -f6 | cut -d'/' -f2)
      printf '
      workspaceRegion='$AMP_WORKSPACE_REGION'
      workspaceId='$AMP_WORKSPACE_ID'
      ' > ../../../../deployments/add-ons/prometheus/params.env
      kustomize build ../../../../deployments/add-ons/prometheus | kubectl apply -f -
    EOT
  }

  depends_on = [aws_prometheus_workspace.kubeflow]
}
