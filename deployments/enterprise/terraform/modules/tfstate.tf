terraform {
  backend "s3" {
    role_arn     = "arn:aws:iam::xxx:role/TerraformServiceRole"
    bucket       = "xxx-ml-ops-tfstate-production"
    key          = "kubeflow/terraform.tfstate"
    region       = "xxx"
    session_name = "terraform"
    encrypt      = true
  }
}
