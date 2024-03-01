#terraform {
#  backend "s3" {
#    role_arn     = "arn:aws:iam::xxx:role/TerraformServiceRole"
#    bucket       = "xxx-kubeflow-user-management-tfstate-xxx"
#    key          = "kubeflow-user-management/terraform.tfstate"
#    region       = "xxx"
#    session_name = "terraform"
#    encrypt      = true
#  }
#}