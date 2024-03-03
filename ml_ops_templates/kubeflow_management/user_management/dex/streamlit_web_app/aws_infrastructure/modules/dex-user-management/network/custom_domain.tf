resource "aws_route53_record" "pre_cognito_domain_a_record" {
  allow_overwrite = true
  zone_id         = aws_route53_zone.env_sub_domain.zone_id
  name            = aws_route53_zone.env_sub_domain.name
  type            = "A"
  ttl             = "300"
  records         = ["127.0.0.1"]

  lifecycle {
    ignore_changes = [records, alias, ttl]
  }
}

resource "aws_acm_certificate" "cognito_domain_cert" {
  domain_name       = "*.${aws_route53_zone.env_sub_domain.name}"
  validation_method = "DNS"
  tags = {
    Name = "${var.ecs_cluster_name}-Certificate"
  }

  lifecycle {
    create_before_destroy = true
  }

  provider = aws.virginia
}

resource "aws_route53_record" "certificate_validation_cognito_domain" {
  for_each = {
    for dvo in aws_acm_certificate.cognito_domain_cert.domain_validation_options : dvo.domain_name => {
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
  zone_id         = aws_route53_zone.env_sub_domain.zone_id
}

resource "aws_acm_certificate_validation" "cognito_domain" {
  provider                = aws.virginia
  certificate_arn         = aws_acm_certificate.cognito_domain_cert.arn
  validation_record_fqdns = [for record in aws_route53_record.certificate_validation_cognito_domain : record.fqdn]
}

resource "aws_cognito_user_pool_domain" "kubeflow" {
  domain          = "auth.${aws_route53_zone.env_sub_domain.name}"
  certificate_arn = aws_acm_certificate.cognito_domain_cert.arn
  user_pool_id    = aws_cognito_user_pool.kubeflow.id

  depends_on = [
    aws_route53_record.pre_cognito_domain_a_record
  ]
}

resource "aws_route53_record" "auth_cognito_domain_record" {
  allow_overwrite = true
  name            = aws_cognito_user_pool_domain.kubeflow.domain
  type            = "A"
  zone_id         = aws_route53_zone.env_sub_domain.zone_id

  alias {
    evaluate_target_health = false
    name                   = aws_cognito_user_pool_domain.kubeflow.cloudfront_distribution_arn
    zone_id                = "Z2FDTNDATAQYW2"
  }
}
