"""

Management of tensorboard setup

"""

import argparse
import subprocess


PARSER = argparse.ArgumentParser(description="manage kubeflow tensorboard")
PARSER.add_argument('-aws_account_id', type=str, required=True, default=None, help='AWS account id')
PARSER.add_argument('-profile_namespace', type=str, required=True, default=None, help='name of the profile namespace')
PARSER.add_argument('-service_account_name', type=str, required=False, default='default-editor', help='name of the service account')
PARSER.add_argument('-aws_region', type=str, required=False, default='eu-central-1', help='AWS region code')
PARSER.add_argument('-cluster_name', type=str, required=False, default='kubeflow', help='name of the EKS cluster')
PARSER.add_argument('-s3_iam_role_policy_name', type=str, required=False, default='AmazonS3FullAccess', help='name of the S3 IAM policy attached to IAM role for service account (IRSA)')
ARGS = PARSER.parse_args()


class KubeflowTensorboardManagementException(Exception):
    """
    Class for handling exceptions from class KubeflowTensorboard
    """
    pass


class KubeflowTensorboard:
    """
    Class for handling Kubeflow tensorboard management
    """
    def __init__(self,
                 aws_account_id: str,
                 profile_namespace: str,
                 service_account_name: str = 'default-editor',
                 aws_region: str = 'eu-central-1',
                 cluster_name: str = 'kubeflow',
                 s3_iam_role_policy_name: str = 'AmazonS3ReadOnlyAccess',
                 ):
        """
        :param aws_account_id: str
            AWS Account ID

        :param profile_namespace: str
            Name of the profile namespace

        :param service_account_name: str
            Name of the service account name

        :param aws_region: str
            AWS region

        :param cluster_name: str
            Name of the EKS cluster

        :param s3_iam_role_policy_name: str
            Name of the S3 IAM role policy
        """
        self.profile_namespace: str = profile_namespace
        self.service_account_name: str = service_account_name
        self.cluster_name: str = cluster_name
        self.aws_region: str = aws_region
        self.aws_account_id: str = aws_account_id
        self.s3_iam_role_policy_name: str = s3_iam_role_policy_name
        self._login_to_eks_cluster()

    def _adjust_s3_poddefault(self) -> None:
        """
        Adjust S3 poddefault yaml
        """
        with open('s3_poddefault.yaml', 'r') as file:
            _s3_poddefault_yaml = file.read()
        _s3_poddefault_yaml = _s3_poddefault_yaml.replace("$(AWS_REGION)", self.aws_region)
        _s3_poddefault_yaml = _s3_poddefault_yaml.replace("$(SERVICE_ACCOUNT_NAME)", self.service_account_name)
        with open('new_s3_poddefault.yaml', 'w') as file:
            file.write(_s3_poddefault_yaml)
        subprocess.run(f'kubectl apply -f new_s3_poddefault.yaml -n {self.profile_namespace}', shell=True, capture_output=False, text=True)

    def _create_irsa(self) -> None:
        """
        Create IAM role for service account (IRSA)
        """
        _cmd: str = f"eksctl create iamserviceaccount --name {self.service_account_name} --namespace {self.profile_namespace} --cluster {self.cluster_name} --region {self.aws_region} --attach-policy-arn=arn:aws:iam::aws:policy/{self.s3_iam_role_policy_name}  --override-existing-serviceaccounts --approve"
        subprocess.run(_cmd, shell=True, capture_output=False, text=True)

    def _login_to_eks_cluster(self) -> None:
        """
        Login to running EKS cluster
        """
        _cmd: str = f"aws eks --region {self.aws_region} update-kubeconfig --name {self.cluster_name}"
        subprocess.run(_cmd, shell=True, capture_output=False, text=True)

    def enable_tensorboard(self) -> None:
        """
        Enable tensorboard by accessing S3 AWS services
        """
        self._create_irsa()
        self._adjust_s3_poddefault()


if __name__ == '__main__':
    _kubeflow_tensorboard_management: KubeflowTensorboard = KubeflowTensorboard(aws_account_id=ARGS.aws_account_id,
                                                                                profile_namespace=ARGS.profile_namespace,
                                                                                service_account_name=ARGS.service_account_name,
                                                                                aws_region=ARGS.aws_region,
                                                                                cluster_name=ARGS.cluster_name,
                                                                                s3_iam_role_policy_name=ARGS.s3_iam_role_policy_name
                                                                                )
    _kubeflow_tensorboard_management.enable_tensorboard()
