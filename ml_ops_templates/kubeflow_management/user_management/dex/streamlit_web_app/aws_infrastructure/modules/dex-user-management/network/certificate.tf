resource "aws_acm_certificate" "this" {
  domain_name       = "*.${data.aws_route53_zone.hosted_zone.name}"
  validation_method = "DNS"

  lifecycle {
    create_before_destroy = true
  }

  tags = {
    Name = "${var.ecs_cluster_name}-Certificate"
  }
}

resource "aws_acm_certificate_validation" "this" {
  certificate_arn         = aws_acm_certificate.this.arn
  validation_record_fqdns = [for record in aws_route53_record.certification : record.fqdn]
}
