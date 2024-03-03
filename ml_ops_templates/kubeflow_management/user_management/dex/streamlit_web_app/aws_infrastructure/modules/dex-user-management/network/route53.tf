data "aws_route53_zone" "root_domain" {
  name         = var.domain_name
  private_zone = false
}

resource "aws_route53_zone" "env_sub_domain" {
  name = "${var.sub_domain_name}.${var.domain_name}"

  tags = {
    Platform = "kubeflow-user-on-aws"
  }

  depends_on = [data.aws_route53_zone.root_domain]
}

resource "aws_route53_record" "env_sub_domain_ns_in_root" {
  name    = aws_route53_zone.env_sub_domain.name
  zone_id = data.aws_route53_zone.root_domain.zone_id
  records = aws_route53_zone.env_sub_domain.name_servers
  type    = "NS"
  ttl     = "30"
}

#resource "aws_route53_record" "kubeflow" {
#  name    = ""
#  type    = "A"
#  zone_id = data.aws_route53_zone.root_domain.id

#  alias {
#    name                   = aws_alb.main.dns_name
#    zone_id                = aws_alb.main.zone_id
#    evaluate_target_health = false
#  }
#}
