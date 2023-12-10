data "aws_route53_zone" "root_domain" {
  name         = var.domain_name
  private_zone = false
}

resource "aws_route53_zone" "subdomain" {
  name = var.sub_domain_name

  tags = {
    Platform = "kubeflow-on-aws"
  }

  depends_on = [data.aws_route53_zone.root_domain]
}

resource "aws_route53_record" "subdomain_ns_in_root" {
  name    = aws_route53_zone.subdomain.name
  zone_id = data.aws_route53_zone.root_domain.zone_id
  records = aws_route53_zone.subdomain.name_servers
  type    = "NS"
  ttl     = "30"
}
