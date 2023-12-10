resource "null_resource" "create_amp_service_account" {
  triggers = {
    always_run = "${timestamp()}"
  }

  provisioner "local-exec" {
    command = <<-EOT
      if kubectl get namespace monitoring &> /dev/null; then
        echo "Namespace monitoring already exists."
      else
        kubectl create namespace monitoring
      fi
      eksctl create iamserviceaccount --name amp-iamproxy-ingest-service-account --namespace monitoring --cluster ${var.cluster_name} --attach-policy-arn ${aws_iam_policy.prometheus.arn} --override-existing-serviceaccounts --approve --region ${var.cluster_region}
    EOT
  }

  depends_on = [aws_iam_role_policy_attachment.prometheus]
}
