terraform {
  backend "s3" {
    role_arn     = "arn:aws:iam::$(AWS_ACCOUNT_ID):role/TerraformServiceRole"
    bucket       = "xxx-ml-ops-tfstate-prod"
    key          = "kubeflow/terraform.tfstate"
    region       = "$(AWS_REGION)"
    session_name = "terraform"
    encrypt      = true
  }
}
