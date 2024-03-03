resource "aws_acm_certificate" "root_domain" {
  domain_name       = "*.${data.aws_route53_zone.root_domain.name}"
  validation_method = "DNS"
  #tags              = var.tags

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_route53_record" "root_domain" {
  for_each = {
    for dvo in aws_acm_certificate.root_domain.domain_validation_options : dvo.domain_name => {
      name   = dvo.resource_record_name
      record = dvo.resource_record_value
      type   = dvo.resource_record_type
    }
  }

  allow_overwrite = true
  name            = each.value.name
  records         = [each.value.record]
  ttl             = 60
  type            = each.value.type
  zone_id         = data.aws_route53_zone.root_domain.zone_id
}

resource "aws_acm_certificate_validation" "root_domain" {
  certificate_arn         = aws_acm_certificate.root_domain.arn
  validation_record_fqdns = [for record in aws_route53_record.root_domain : record.fqdn]
}
