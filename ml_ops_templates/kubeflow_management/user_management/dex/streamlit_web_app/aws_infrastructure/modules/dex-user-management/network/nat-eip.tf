data "aws_availability_zones" "available" {}

locals {
  availability_zone_names = data.aws_availability_zones.available.names
}

resource "aws_eip" "nat" {
  count = 1

  tags = {
    Name = "Kubeflow User Management-EIP"
  }
}

resource "aws_nat_gateway" "nat" {
  count         = 1
  subnet_id     = aws_subnet.public[0].id
  allocation_id = aws_eip.nat[0].id

  tags = {
    Name = "Kubeflow User Management-NAT Gateway"
  }
}
