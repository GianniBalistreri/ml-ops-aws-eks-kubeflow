terraform {
  backend "s3" {
    role_arn     = "arn:aws:iam::711117404296:role/TerraformServiceRole"
    bucket       = "shopware-ml-ops-tfstate-production"
    key          = "kubeflow/terraform.tfstate"
    region       = "eu-central-1"
    session_name = "terraform"
    encrypt      = true
  }
}
