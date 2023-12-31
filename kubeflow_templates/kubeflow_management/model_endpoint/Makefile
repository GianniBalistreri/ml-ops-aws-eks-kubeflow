
add-domain:
	python kubeflow_model_endpoint_management.py -aws_account_id=973450839031 -profile_namespace=production -top_level_domain_name=com -second_level_domain_name=shopware-kubeflow -subdomain_name=dev -service_account_name=default-editor -aws_region=eu-central-1 -cluster_name=kubeflow -ecr_iam_role_policy_name=AmazonEC2ContainerRegistryReadOnly -s3_iam_role_policy_name=AmazonS3ReadOnlyAccess -meth=add_domain

enable-inference-service:
	python kubeflow_model_endpoint_management.py -aws_account_id=<> -profile_namespace=<> -top_level_domain_name=<> -second_level_domain_name=<> -subdomain_name=<> -service_account_name=<> -aws_region=eu-central-1 -cluster_name=kubeflow -ecr_iam_role_policy_name=AmazonEC2ContainerRegistryReadOnly -s3_iam_role_policy_name=AmazonS3ReadOnlyAccess -meth=enable_inference_service
