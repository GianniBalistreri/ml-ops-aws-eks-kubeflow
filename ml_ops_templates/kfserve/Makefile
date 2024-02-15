SHELL := /bin/bash

install-kubectl:
	$(eval KUBECTL_VERSION:=v1.25.0)
	curl -LO "https://dl.k8s.io/release/$(KUBECTL_VERSION)/bin/linux/amd64/kubectl"
	sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
	rm kubectl
	kubectl version --client

deploy:
	kubectl apply -f kfserving-inference.yaml

delete:
	kubectl delete inferenceservice endpoint-name
