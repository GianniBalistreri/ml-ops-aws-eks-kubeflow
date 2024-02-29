resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = var.ecs_enable_dns_hostnames
  enable_dns_support   = true

  tags = {
    Name = "Kubeflow User Management"
  }
}

resource "aws_internet_gateway" "ig" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "Kubeflow User Management"
  }
}
