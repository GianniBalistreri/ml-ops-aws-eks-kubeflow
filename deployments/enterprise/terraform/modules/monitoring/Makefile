
deploy:
	python kiali_monitoring.py -aws_account_id=<> -secret_name=kiali -meth=deploy

delete:
	python kiali_monitoring.py -aws_account_id=<> -meth=delete

validate-logs:
	export kp=$(kubectl -n istio-system get pod -l app=kiali -o jsonpath='{.items[0].metadata.name}')
	kubectl logs -f $kp -n istio-system

port-forward:
	export ks=$(kubectl -n istio-system get service -l app=kiali -o jsonpath='{.items[0].metadata.name}')
	kubectl port-forward -n istio-system svc/$ks 20001:20001
