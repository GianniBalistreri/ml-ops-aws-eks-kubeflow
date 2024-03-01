terraform {
  backend "s3" {
    role_arn     = "arn:aws:iam::xxx:role/TerraformServiceRole"
    bucket       = "xxx-ml-ops-tfstate-xxx"
    key          = "kubeflow/terraform.tfstate"
    region       = "xxx"
    session_name = "terraform"
    encrypt      = true
  }
}
