
add-profile:
	python kubeflow_user_management.py -user_email=<> -user_name=<> -aws_account_id=<> -aws_region=eu-central-1 -include_profile=1 -cluster_name=kubeflow -iam_role_name=kubeflow-user-role -meth=add_profile

add-user:
	python kubeflow_user_management.py -user_email=<> -user_name=<> -aws_account_id=<> -aws_region=eu-central-1 -include_profile=1 -cluster_name=kubeflow -iam_role_name=kubeflow-user-role -meth=add_user -pwd=<> -hashed=0

delete-profile:
	python kubeflow_user_management.py -user_email=<> -user_name=<> -aws_account_id=<> -aws_region=eu-central-1 -include_profile=1 -cluster_name=kubeflow -iam_role_name=kubeflow-user-role -meth=delete_profile

delete-user:
	python kubeflow_user_management.py -user_email=<> -user_name=<> -aws_account_id=<> -aws_region=eu-central-1 -include_profile=1 -cluster_name=kubeflow -iam_role_name=kubeflow-user-role -meth=delete_user

initial-deployment:
	python kubeflow_user_management.py -user_email=<> -user_name=production -aws_account_id=<> -aws_region=eu-central-1 -include_profile=1 -cluster_name=kubeflow -iam_role_name=kubeflow-user-role -meth=initial_deployment -pwd=<> -hashed=0
