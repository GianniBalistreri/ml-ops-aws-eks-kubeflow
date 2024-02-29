output "kubeflow_platform_domain" {
  value = "${var.sub_domain_name}.${var.domain_name}"
}

output "user_pool_arn" {
  description = "Cognito User Pool ARN"
  value       = aws_cognito_user_pool.kubeflow.arn
}

output "app_client_id" {
  description = "Cognito App client Id"
  value       = aws_cognito_user_pool_client.kubeflow.id
}

output "user_pool_domain" {
  description = "Cognito User Pool Domain"
  value       = aws_cognito_user_pool_domain.kubeflow.domain
}

output "logout_url" {
  description = "Logout URL"
  value       = "https://auth.${data.aws_route53_zone.subdomain.name}/logout?client_id=${aws_cognito_user_pool_client.kubeflow.user_pool_id}&logout_uri=https://${data.aws_route53_zone.subdomain.name}"
}
