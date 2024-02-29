resource "aws_subnet" "public" {
  count                   = length(data.aws_availability_zones.available.names)
  vpc_id                  = aws_vpc.main.id
  cidr_block              = var.public_subnet_cidrs[count.index]
  availability_zone       = element(data.aws_availability_zones.available.names, count.index)
  map_public_ip_on_launch = false

  tags = {
    Name = "Tracking Gateway-Public Subnet-${element(data.aws_availability_zones.available.names, count.index)}"
  }
}

resource "aws_subnet" "private" {
  count             = length(data.aws_availability_zones.available.names)
  vpc_id            = aws_vpc.main.id
  cidr_block        = var.private_subnet_cidrs[count.index]
  availability_zone = element(data.aws_availability_zones.available.names, count.index)

  tags = {
    Name = "Tracking Gateway-Private Subnet-${element(data.aws_availability_zones.available.names, count.index)}"
  }
}

resource "aws_route_table" "public" {
  count  = length(data.aws_availability_zones.available.names)
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "Tracking Gateway-Public Route Table-${element(data.aws_availability_zones.available.names, count.index)}"
  }
}

resource "aws_route_table" "private" {
  count  = length(data.aws_availability_zones.available.names)
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "Tracking Gateway-Private Route Table-${element(data.aws_availability_zones.available.names, count.index)}"
  }
}

resource "aws_route_table_association" "private" {
  count          = length(data.aws_availability_zones.available.names)
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = element(aws_route_table.private.*.id, count.index)
}

resource "aws_route_table_association" "public" {
  count          = length(data.aws_availability_zones.available.names)
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = element(aws_route_table.public.*.id, count.index)
}

resource "aws_route" "public" {
  count                  = length(data.aws_availability_zones.available.names)
  route_table_id         = element(aws_route_table.public[*].id, count.index)
  gateway_id             = aws_internet_gateway.ig.id
  destination_cidr_block = "0.0.0.0/0"
}

resource "aws_route" "private" {
  count                  = length(data.aws_availability_zones.available.names)
  route_table_id         = element(aws_route_table.private[*].id, count.index)
  nat_gateway_id         = element(aws_nat_gateway.nat[*].id, count.index)
  destination_cidr_block = "0.0.0.0/0"
}
