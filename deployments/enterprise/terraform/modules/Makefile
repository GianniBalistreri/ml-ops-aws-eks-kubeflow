SHELL := /bin/bash # Use bash syntax

deploy: init \
 		create-vpc \
 		create-eks-cluster \
 		deploy-eks-blueprints-k8s-addons \
 		deploy-kubeflow-components \
 		deploy-monitoring

delete: delete-monitoring \
		delete-kubeflow-components \
 		delete-eks-blueprints-k8s-addons \
 		delete-eks-cluster \
 		delete-vpc
		terraform destroy -auto-approve

deploy-eks: init \
 			create-vpc \
 			create-eks-cluster \
 			deploy-eks-blueprints-k8s-addons

init:
	terraform init


create-vpc:
	terraform apply -target="module.vpc" -auto-approve

create-eks-cluster:
	terraform apply -target="module.eks_blueprints" -auto-approve
	terraform output -raw configure_kubectl | bash

deploy-eks-blueprints-k8s-addons:
	terraform apply -target="module.eks_blueprints_kubernetes_addons" -auto-approve

deploy-kubeflow-components:
	terraform apply -target="module.kubeflow_components" -auto-approve

deploy-monitoring:
	terraform apply -target="module.monitoring" -auto-approve

deploy-load-balancer:
	terraform apply -target="module.load_balancer" -auto-approve

deploy-custom-components:
	terraform apply -target="module.custom_components" -auto-approve


delete-vpc:
	terraform destroy -target="module.vpc" -auto-approve

delete-eks-cluster:
	terraform destroy -target="module.eks_blueprints" -auto-approve

delete-eks-blueprints-k8s-addons:
	terraform destroy -target="module.eks_blueprints_kubernetes_addons" -auto-approve

delete-kubeflow-components:
	terraform destroy -target="module.kubeflow_components" -auto-approve

delete-monitoring:
	terraform destroy -target="module.monitoring" -auto-approve

delete-load-balancer:
	terraform destroy -target="module.load_balancer" -auto-approve

delete-custom-components:
	terraform destroy -target="module.custom_components" -auto-approve

# don't create executables
.PHONY: *