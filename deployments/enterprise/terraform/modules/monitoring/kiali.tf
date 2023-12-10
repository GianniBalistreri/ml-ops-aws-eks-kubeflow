resource "null_resource" "kiali" {
  triggers = {
    always_run = "${timestamp()}"
  }

  provisioner "local-exec" {
    command = <<EOT
      kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.19/samples/addons/prometheus.yaml
      kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.19/samples/addons/grafana.yaml
      kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.19/samples/addons/jaeger.yaml
      helm repo add kiali https://kiali.org/helm-charts
      helm repo update
      helm install \
        --set cr.create=true \
        --set cr.namespace=istio-system \
        --namespace kiali-operator \
        --create-namespace \
        kiali-operator \
        kiali/kiali-operator
    EOT
  }

  depends_on = [aws_grafana_workspace.kubeflow]
}