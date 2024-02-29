#terraform {
#  backend "s3" {
#    role_arn     = "arn:aws:iam::711117404296:role/TerraformServiceRole"
#    bucket       = "shopware-tracking-gateway-tfstate-production"
#    key          = "tracking-gateway/terraform.tfstate"
#    region       = "eu-central-1"
#    session_name = "terraform"
#    encrypt      = true
#  }
#}