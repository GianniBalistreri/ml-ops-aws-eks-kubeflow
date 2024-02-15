data "aws_route53_zone" "root_domain" {
  name         = "${var.domain_name}.${var.top_level_domain_name}"
  private_zone = false
}

resource "aws_route53_zone" "env_sub_domain" {
  name = "${var.environment_sub_domain_name}.${var.domain_name}.${var.top_level_domain_name}"

  tags = {
    Platform = "kubeflow-on-aws"
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

resource "aws_route53_zone" "namespace_sub_domain" {
  name = "${var.namespace_sub_domain_name}.${var.environment_sub_domain_name}.${var.domain_name}.${var.top_level_domain_name}"

  tags = {
    Platform = "kubeflow-on-aws"
  }

  depends_on = [data.aws_route53_zone.root_domain]
}

resource "aws_route53_record" "namespace_sub_domain_ns_in_root" {
  name    = aws_route53_zone.namespace_sub_domain.name
  zone_id = data.aws_route53_zone.root_domain.zone_id
  records = aws_route53_zone.namespace_sub_domain.name_servers
  type    = "NS"
  ttl     = "30"
}
