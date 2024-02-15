data "aws_route53_zone" "sub_domain" {
  name = "${var.environment_sub_domain_name}.${var.domain_name}.${var.top_level_domain_name}"
}

data "aws_acm_certificate" "sub_domain" {
  domain = "*.${var.environment_sub_domain_name}.${var.domain_name}.${var.top_level_domain_name}"
}

resource "null_resource" "istio_ingress" {
  triggers = {
    always_run = "${timestamp()}"
  }

  provisioner "local-exec" {
    command = <<EOT
      printf '
      certArn='${data.aws_acm_certificate.sub_domain.arn}'
      httpHeaderName='${var.http_header_name}'
      httpHeaderValues='["token1", "token2"]'
      ' > ../../../../awsconfigs/common/istio-ingress/overlays/api/params.env
      printf 'certArn='${data.aws_acm_certificate.sub_domain.arn}'' > ../../../../awsconfigs/common/istio-ingress/overlays/https/params.env
      printf 'loadBalancerScheme='${var.load_balancer_scheme}'' > ../../../../awsconfigs/common/istio-ingress/base/params.env
      kustomize build ../../../../awsconfigs/common/istio-ingress/overlays/https | kubectl apply -f -
      sleep 180
    EOT
  }
}

data "aws_lb" "istio_ingress" {
  tags = {
    "elbv2.k8s.aws/cluster" = var.cluster_name
    "ingress.k8s.aws/stack" = "istio-system/istio-ingress"
  }

  depends_on = [null_resource.istio_ingress]
}

resource "aws_route53_record" "cname" {
  name    = "*.${var.environment_sub_domain_name}.${var.domain_name}.${var.top_level_domain_name}"
  type    = "CNAME"
  zone_id = data.aws_route53_zone.sub_domain.zone_id
  records = [data.aws_lb.istio_ingress.dns_name]
  ttl     = "300"

  depends_on = [data.aws_lb.istio_ingress]
}
