resource "aws_security_group" "ecs_fargate" {
  name   = "elasticache-sg"
  vpc_id = var.vpc_id

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_security_group_rule" "ecs_fargate_egress" {
  type              = "egress"
  from_port         = 0
  to_port           = 0
  protocol          = "-1"
  security_group_id = aws_security_group.ecs_fargate.id
  cidr_blocks       = ["0.0.0.0/0"]

  depends_on = [aws_security_group.ecs_fargate]
}

resource "aws_security_group_rule" "ecs_fargate_ingress" {
  type              = "ingress"
  from_port         = 6379
  to_port           = 6379
  protocol          = "tcp"
  security_group_id = aws_security_group.ecs_fargate.id
  cidr_blocks       = var.public_subnet_cidrs

  depends_on = [aws_security_group.ecs_fargate]
}

resource "aws_elasticache_cluster" "ecs_fargate" {
  cluster_id           = "elasticache-cluster"
  engine               = "redis"
  node_type            = "cache.t4g.micro"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  engine_version       = "7.0"
  security_group_ids   = [aws_security_group.ecs_fargate.id]
  subnet_group_name    = aws_elasticache_subnet_group.elasticache.name
  port                 = 6379
}

resource "aws_elasticache_subnet_group" "elasticache" {
  name       = "elasticache-subnet-group"
  subnet_ids = var.subnet_ids
}
