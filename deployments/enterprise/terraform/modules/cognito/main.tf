data "aws_acm_certificate" "subdomain" {
  domain = "*.${var.sub_domain_name}.${var.domain_name}.${var.top_level_domain_name}"
}

resource "null_resource" "istio_ingress" {
  triggers = {
    always_run = "${timestamp()}"
  }

  provisioner "local-exec" {
    command = <<EOT
      printf '
      CognitoUserPoolArn='${aws_cognito_user_pool.kubeflow.arn}'
      CognitoAppClientId='${aws_cognito_user_pool_client.kubeflow.id}'
      CognitoUserPoolDomain='auth.prod.shopware-kubeflow.io'
      certArn='${data.aws_acm_certificate.subdomain.arn}'
      ' > ../../../../awsconfigs/common/istio-ingress/overlays/cognito/params.env
      printf 'loadBalancerScheme='${var.load_balancer_scheme}'' > ../../../../awsconfigs/common/istio-ingress/base/params.env
      kustomize build ../../../../awsconfigs/common/istio-ingress/overlays/cognito | kubectl apply -f -
      sleep 180
    EOT
  }

  depends_on = [aws_route53_record.auth_cognito_domain_record]
}

# Implement ingress in terraform instead of using chart to use features like wait_for_load_balancer
#resource "kubernetes_ingress_v1" "istio_ingress" {
#  wait_for_load_balancer = true

#  metadata {
#    annotations = {
#      "alb.ingress.kubernetes.io/auth-type" : "cognito",
#      "alb.ingress.kubernetes.io/auth-idp-cognito" : "{\"UserPoolArn\":\"${aws_cognito_user_pool.platform.arn}\",\"UserPoolClientId\":\"${aws_cognito_user_pool_client.platform.id}\", \"UserPoolDomain\":\"${aws_cognito_user_pool.platform.domain}\"}"
#      "alb.ingress.kubernetes.io/certificate-arn" : "${data.aws_acm_certificate.subdomain.arn}"
#      "alb.ingress.kubernetes.io/listen-ports" : "[{\"HTTPS\":443}]",
#      "alb.ingress.kubernetes.io/target-type" : "ip",
#      "alb.ingress.kubernetes.io/load-balancer-attributes" : "routing.http.drop_invalid_header_fields.enabled=true",
#      "alb.ingress.kubernetes.io/actions.istio-ingressgateway": "{\"Type\":\"forward\",\"ForwardConfig\":{\"TargetGroups\":[{\"ServiceName\":\"istio-ingressgateway\",\"ServicePort\":\"80\",\"Weight\":100}]}}"
#      "alb.ingress.kubernetes.io/scheme" : "${var.load_balancer_scheme}"
#      "alb.ingress.kubernetes.io/tags" : trim(trimspace(replace(replace(jsonencode(var.tags), "\"", ""), ":", "=")), "{}")
#    }
#    name      = "istio-ingress"
#    namespace = "istio-system"
#  }

#  spec {
#    ingress_class_name = "alb"
#    rule {
#      http {
#        path {
#          path = "/*"
#          backend {
#            service {
#              name = "istio-ingressgateway"
#              port {
#                number = 80
#              }
#            }
#          }
#          path_type = "ImplementationSpecific"
#        }
#      }
#    }
#  }

#  depends_on = [aws_route53_record.auth_cognito_domain_record]
#}

data "aws_lb" "istio_ingress" {
  tags = {
    "elbv2.k8s.aws/cluster" = var.cluster_name
    "ingress.k8s.aws/stack" = "istio-system/istio-ingress"
  }
  depends_on = [null_resource.istio_ingress]
}

resource "aws_route53_record" "cname_record" {
  allow_overwrite = true
  name            = "${var.second_sub_domain_name}.${data.aws_route53_zone.subdomain.name}"
  zone_id         = data.aws_route53_zone.subdomain.zone_id
  type            = "CNAME"
  records         = [data.aws_lb.istio_ingress.dns_name]
  ttl             = "300"

  depends_on = [data.aws_lb.istio_ingress]
}

resource "aws_route53_record" "a_record" {
  allow_overwrite = true
  zone_id         = data.aws_route53_zone.subdomain.zone_id
  name            = data.aws_route53_zone.subdomain.name
  type            = "A"

  alias {
    name                   = data.aws_lb.istio_ingress.dns_name
    zone_id                = data.aws_lb.istio_ingress.zone_id
    evaluate_target_health = false
  }

  depends_on = [data.aws_lb.istio_ingress]
}
