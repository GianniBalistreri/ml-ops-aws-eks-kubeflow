create-ingress:
	certArn=$(aws acm list-certificates --query "CertificateSummaryList[?DomainName=='*.platform.shopware-kubeflow.io']" --output json | jq -r '.[0].CertificateArn')
	printf 'certArn='$certArn'' > ../../../../awsconfigs/common/istio-ingress/overlays/https/params.env
    printf 'clusterName=kubeflow' > ../../../../awsconfigs/common/aws-alb-ingress-controller/base/params.env
    helm delete aws-load-balancer-controller -n kube-system
	kustomize build ../../../../deployments/add-ons/load-balancer | kubectl apply -f -
	#kustomize build ../../../../awsconfigs/common/istio-ingress/overlays/https | kubectl apply -f -

assign-alb-ingress-address:
	kubectl -n istio-system get service istio-ingressgateway --output jsonpath={.status.loadBalancer.ingress[0].hostname}
	# kubectl -n istio-system get service istio-ingressgateway --output jsonpath={.status.loadBalancer.ingress[0].ip}
	kubectl apply -f  tls-manifest/certificate.yaml

delete-ingress:
	kubectl delete deployment istio-ingress -n istio-system


deploy: create-ingress \
        assign-alb-ingress-address
