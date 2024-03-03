resource "aws_route53_record" "cname_record" {
  allow_overwrite = true
  zone_id         = aws_route53_zone.env_sub_domain.zone_id
  name            = "${var.callback_logout_sub_domain_name}.${aws_route53_zone.env_sub_domain.name}"
  type            = "CNAME"
  records         = [aws_alb.main.dns_name]
  ttl             = "300"
}

resource "aws_route53_record" "a_record" {
  allow_overwrite = true
  zone_id         = aws_route53_zone.env_sub_domain.zone_id
  name            = aws_route53_zone.env_sub_domain.name
  type            = "A"

  alias {
    name                   = aws_alb.main.dns_name
    zone_id                = aws_alb.main.zone_id
    evaluate_target_health = false
  }
}
