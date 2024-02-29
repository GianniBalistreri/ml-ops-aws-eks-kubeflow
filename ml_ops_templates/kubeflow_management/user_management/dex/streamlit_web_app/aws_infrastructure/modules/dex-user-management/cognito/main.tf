data "aws_acm_certificate" "subdomain" {
  domain = "*.${var.sub_domain_name}.${var.domain_name}"
}

resource "aws_route53_record" "cname_record" {
  allow_overwrite = true
  name            = "${var.sub_domain_name}.${var.domain_name}"
  zone_id         = data.aws_route53_zone.subdomain.zone_id
  type            = "CNAME"
  records         = [var.alb_dns_name]
  ttl             = "300"
}

resource "aws_route53_record" "a_record" {
  allow_overwrite = true
  zone_id         = data.aws_route53_zone.subdomain.zone_id
  name            = data.aws_route53_zone.subdomain.name
  type            = "A"

  alias {
    name                   = var.alb_dns_name
    zone_id                = var.alb_zone_id
    evaluate_target_health = false
  }
}
