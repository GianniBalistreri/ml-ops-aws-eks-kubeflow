terraform {
  backend "s3" {
    role_arn     = "arn:aws:iam::AWS_ACCOUNT_ID:role/TerraformServiceRole"
    bucket       = "xxx-ml-ops-tfstate-production"
    key          = "kubeflow/terraform.tfstate"
    region       = "eu-central-1"
    session_name = "terraform"
    encrypt      = true
  }
}
