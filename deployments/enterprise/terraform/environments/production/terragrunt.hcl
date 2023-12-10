remote_state {
  backend = "s3"

  generate = {
    path      = "backend.tf"
    if_exists = "overwrite_terragrunt"
  }

  config = {
    role_arn     = "arn:aws:iam::AWS_ACCOUNT_ID:role/TerraformServiceRole"
    bucket       = "xxx-ml-ops-tfstate-production"
    key          = "kubeflow/terraform.tfstate"
    region       = "eu-central-1"
    session_name = "terraform"
    encrypt      = true
    #dynamodb_table = "terraform-locks"
  }
}