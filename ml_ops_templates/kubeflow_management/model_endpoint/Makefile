
add-domain:
	python kubeflow_model_endpoint_management.py -s3_endpoint=s3.eu-central-1.amazonaws.com -profile_namespace=production -top_level_domain_name=net -second_level_domain_name=shopware-kubeflow -subdomain_name=dev -service_account_name=sa -aws_region=eu-central-1 -cluster_name=kubeflow -ecr_iam_role_policy_name=AmazonEC2ContainerRegistryReadOnly -s3_iam_role_policy_name=AmazonS3ReadOnlyAccess -meth=add_domain

enable-inference-service:
	python kubeflow_model_endpoint_management.py -s3_endpoint=s3.eu-central-1.amazonaws.com -profile_namespace=production -top_level_domain_name=net -second_level_domain_name=shopware-kubeflow -subdomain_name=dev -service_account_name=sa -aws_region=eu-central-1 -cluster_name=kubeflow -ecr_iam_role_policy_name=AmazonEC2ContainerRegistryReadOnly -s3_iam_role_policy_name=AmazonS3ReadOnlyAccess -meth=enable_inference_service
