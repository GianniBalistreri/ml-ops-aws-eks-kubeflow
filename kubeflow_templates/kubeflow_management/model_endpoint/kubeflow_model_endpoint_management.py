"""

Management of model endpoints in kserve & knative

"""

import argparse
import subprocess

from typing import List


PARSER = argparse.ArgumentParser(description="manage kubeflow model endpoint")
PARSER.add_argument('-aws_account_id', type=str, required=True, default=None, help='AWS account id')
PARSER.add_argument('-profile_namespace', type=str, required=True, default=None, help='name of the profile namespace')
PARSER.add_argument('-top_level_domain_name', type=str, required=True, default=None, help='name of the top level domain')
PARSER.add_argument('-second_level_domain_name', type=str, required=True, default=None, help='name of the second level domain')
PARSER.add_argument('-subdomain_name', type=str, required=True, default=None, help='name of the subdomain')
PARSER.add_argument('-service_account_name', type=str, required=False, default='default-editor', help='name of the service account')
PARSER.add_argument('-aws_region', type=str, required=False, default='eu-central-1', help='AWS region code')
PARSER.add_argument('-cluster_name', type=str, required=False, default='kubeflow', help='name of the EKS cluster')
PARSER.add_argument('-ecr_iam_role_policy_name', type=str, required=False, default='AmazonEC2ContainerRegistryReadOnly', help='name of the ECR IAM policy attached to IAM role for service account (IRSA)')
PARSER.add_argument('-s3_iam_role_policy_name', type=str, required=False, default='AmazonS3ReadOnlyAccess', help='name of the S3 IAM policy attached to IAM role for service account (IRSA)')
PARSER.add_argument('-meth', type=str, required=True, default=None, help='method name of class KubeflowModelEndpointManagement to apply')
ARGS = PARSER.parse_args()


class KubeflowModelEndpointManagementException(Exception):
    """
    Class for handling exceptions from class KubeflowModelEndpointManagement
    """
    pass


class KubeflowModelEndpointManagement:
    """
    Class for handling Kubeflow model endpoint management
    """
    def __init__(self,
                 aws_account_id: str,
                 profile_namespace: str,
                 top_level_domain_name: str,
                 second_level_domain_name: str,
                 subdomain_name: str = None,
                 service_account_name: str = 'default-editor',
                 aws_region: str = 'eu-central-1',
                 cluster_name: str = 'kubeflow',
                 ecr_iam_role_policy_name: str = 'AmazonEC2ContainerRegistryReadOnly',
                 s3_iam_role_policy_name: str = 'AmazonS3ReadOnlyAccess',
                 ):
        """
        :param aws_account_id: str
            AWS Account ID

        :param profile_namespace: str
            Name of the profile namespace

        :param top_level_domain_name: str
            Name of the top level domain

        :param second_level_domain_name: str
            Name of the second level domain

        :param subdomain_name: str
            Name of the subdomain

        :param service_account_name: str
            Name of the service account

        :param aws_region: str
            AWS region

        :param cluster_name: str
            Name of the EKS cluster

        :param ecr_iam_role_policy_name: str
            Name of the ECR IAM role policy

        :param s3_iam_role_policy_name: str
            Name of the S3 IAM role policy
        """
        self.domain_name: str = f'{subdomain_name}.{second_level_domain_name}.{top_level_domain_name}'
        self.profile_namespace: str = profile_namespace
        self.service_account_name: str = service_account_name
        self.cluster_name: str = cluster_name
        self.aws_region: str = aws_region
        self.aws_account_id: str = aws_account_id
        self.ecr_iam_role_policy_name: str = ecr_iam_role_policy_name
        self.s3_iam_role_policy_name: str = s3_iam_role_policy_name
        self._login_to_eks_cluster()

    def _adjust_default_domain_knative_services(self) -> str:
        """
        Adjust default domain for all knative services

        :return: str
            Adjusted knative service configmap yaml file
        """
        _cmd: str = "kubectl get configmap config-domain -n knative-serving -o yaml > knative_service_configmap.yaml"
        subprocess.run(_cmd, shell=True, capture_output=False, text=True)
        with open('knative_service_configmap.yaml', 'r') as file:
            _knative_service_configmap_yaml = file.read()
        _config_yaml: List[str] = []
        _ignore_line: bool = False
        for line in _knative_service_configmap_yaml.split('\n'):
            if line.find('_example: |') >= 0:
                _ignore_line = True
                _config_yaml.append(f'  {self.domain_name}: ""')
            if line.find('kind: ConfigMap') >= 0:
                _ignore_line = False
            if not _ignore_line:
                _config_yaml.append(line)
        return "\n".join(_config_yaml)

    def _attach_secret_to_irsa(self) -> None:
        """
        Attach secret to IRSA in profile namespace
        """
        _p: str = '{"secrets": [{"name": "aws-secret"}]}'
        _cmd: str = f"kubectl patch serviceaccount {self.service_account_name} -n {self.profile_namespace} -p '{_p}'"
        subprocess.run(_cmd, shell=True, capture_output=False, text=True)

    def _create_irsa(self) -> None:
        """
        Create IAM role for service account (IRSA)
        """
        _cmd: str = f"eksctl create iamserviceaccount --name {self.service_account_name} --namespace {self.profile_namespace} --cluster {self.cluster_name} --region {self.aws_region} --attach-policy-arn=arn:aws:iam::aws:policy/{self.ecr_iam_role_policy_name} --attach-policy-arn=arn:aws:iam::aws:policy/{self.s3_iam_role_policy_name}  --override-existing-serviceaccounts --approve"
        subprocess.run(_cmd, shell=True, capture_output=False, text=True)

    def _create_secret(self) -> None:
        """
        Create secret
        """
        with open('add_secret.yaml', 'r') as file:
            _secret_yaml = file.read()
        _secret_yaml = _secret_yaml.replace("$(PROFILE_NAME)", self.profile_namespace)
        _secret_yaml = _secret_yaml.replace("$(CLUSTER_REGION)", self.aws_region)
        with open('new_secret.yaml', 'w') as file:
            file.write(_secret_yaml)
        subprocess.run('kubectl apply -f new_secret.yaml', shell=True, capture_output=False, text=True)

    def _login_to_eks_cluster(self) -> None:
        """
        Login to running EKS cluster
        """
        _cmd: str = f"aws eks --region {self.aws_region} update-kubeconfig --name {self.cluster_name}"
        subprocess.run(_cmd, shell=True, capture_output=False, text=True)

    def add_domain(self) -> None:
        """
        Add custom domain to knative services
        """
        _adjusted_knative_service_configmap_yaml: str = self._adjust_default_domain_knative_services()
        with open('new_knative_service_configmap.yaml', 'w') as file:
            file.write(_adjusted_knative_service_configmap_yaml)
        subprocess.run('kubectl apply -f new_knative_service_configmap.yaml', shell=True, capture_output=False, text=True)

    def enable_inference_service(self) -> None:
        """
        Enable inference service by accessing ECR and S3 AWS services
        """
        self._create_irsa()
        self._create_secret()
        self._attach_secret_to_irsa()


if __name__ == '__main__':
    _kubeflow_model_endpoint_management: KubeflowModelEndpointManagement = KubeflowModelEndpointManagement(aws_account_id=ARGS.aws_account_id,
                                                                                                           profile_namespace=ARGS.profile_namespace,
                                                                                                           top_level_domain_name=ARGS.top_level_domain_name,
                                                                                                           second_level_domain_name=ARGS.second_level_domain_name,
                                                                                                           subdomain_name=ARGS.subdomain_name,
                                                                                                           service_account_name=ARGS.service_account_name,
                                                                                                           aws_region=ARGS.aws_region,
                                                                                                           cluster_name=ARGS.cluster_name,
                                                                                                           ecr_iam_role_policy_name=ARGS.ecr_iam_role_policy_name,
                                                                                                           s3_iam_role_policy_name=ARGS.s3_iam_role_policy_name
                                                                                                           )
    if ARGS.meth == 'add_domain':
        _kubeflow_model_endpoint_management.add_domain()
    elif ARGS.meth == 'enable_inference_service':
        _kubeflow_model_endpoint_management.enable_inference_service()
    else:
        raise KubeflowModelEndpointManagementException(f'Method ({ARGS.meth}) not supported')
