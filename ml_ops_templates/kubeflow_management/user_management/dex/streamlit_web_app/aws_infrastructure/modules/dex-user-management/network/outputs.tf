output "vpc_id" {
  value = aws_vpc.main.id
}

output "vpc_cidr_block" {
  value = aws_vpc.main.cidr_block
}

output "public_subnets_id" {
  value = aws_subnet.public.*.id
}

output "private_subnets_id" {
  value = aws_subnet.private.*.id
}

output "domain_name" {
  value = aws_route53_record.kubeflow.name
}

output "ecs_task_id" {
  value = aws_security_group.ecs_tasks.id
}

output "alb_target_group_arn" {
  value = aws_alb_target_group.kubeflow.arn
}

output "alb_dns_name" {
  value = aws_alb.main.dns_name
}

output "alb_zone_id" {
  value = aws_alb.main.zone_id
}

output "subnet_ids" {
  value = [for subnet in aws_subnet.private : subnet.id]
}
