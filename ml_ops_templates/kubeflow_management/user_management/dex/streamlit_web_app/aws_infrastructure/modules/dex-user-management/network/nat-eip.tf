data "aws_availability_zones" "available" {}

locals {
  availability_zone_names = data.aws_availability_zones.available.names
}

resource "aws_eip" "nat" {
  count = length(data.aws_availability_zones.available.names)

  tags = {
    Name = "Tracking Gateway-EIP-${element(local.availability_zone_names, count.index)}"
  }
}

resource "aws_nat_gateway" "nat" {
  count         = length(data.aws_availability_zones.available.names)
  subnet_id     = element(aws_subnet.public.*.id, count.index)
  allocation_id = element(aws_eip.nat.*.id, count.index)

  tags = {
    Name = "Tracking Gateway-NAT gateway-${element(local.availability_zone_names, count.index)}"
  }
}
