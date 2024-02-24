resource "aws_cognito_user_pool_client" "kubeflow" {
  name                                 = var.cluster_name
  user_pool_id                         = aws_cognito_user_pool.kubeflow.id
  generate_secret                      = true
  callback_urls                        = ["https://${var.callback_logout_sub_domain_name}.${var.environment_sub_domain_name}.${var.domain_name}.${var.top_level_domain_name}/oauth2/idpresponse"]
  logout_urls                          = ["https://${var.callback_logout_sub_domain_name}.${var.environment_sub_domain_name}.${var.domain_name}.${var.top_level_domain_name}"]
  allowed_oauth_flows_user_pool_client = true
  allowed_oauth_flows                  = var.cognito_user_pool_client_allowed_oauth_flows
  allowed_oauth_scopes                 = ["email", "openid", "profile", "aws.cognito.signin.user.admin"]
  supported_identity_providers         = ["COGNITO"]
}
