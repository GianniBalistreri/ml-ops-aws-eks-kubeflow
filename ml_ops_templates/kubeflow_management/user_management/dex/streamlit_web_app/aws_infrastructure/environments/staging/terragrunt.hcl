remote_state {
  backend = "s3"

  generate = {
    path      = "backend.tf"
    if_exists = "overwrite_terragrunt"
  }

  config = {
    role_arn     = "arn:aws:iam::xxx:role/TerraformServiceRole"
    bucket       = "xxx-kubeflow-user-management-tfstate-staging"
    key          = "kubeflow-user-management/terraform.tfstate"
    region       = "xxx"
    session_name = "terraform"
    encrypt      = true
    #dynamodb_table = "terraform-locks"
  }
}
