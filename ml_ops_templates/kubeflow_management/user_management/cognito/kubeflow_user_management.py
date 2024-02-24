"""

Management of Kubeflow users in AWS Cognito without using external identity provider service

"""

import argparse
import subprocess


PARSER = argparse.ArgumentParser(description="manage kubeflow user")
PARSER.add_argument('-user_email', type=str, required=True, default=None, help='user email address')
PARSER.add_argument('-user_name', type=str, required=True, default=None, help='user name')
PARSER.add_argument('-aws_account_id', type=str, required=True, default=None, help='AWS account id')
PARSER.add_argument('-aws_region', type=str, required=False, default='eu-central-1', help='AWS region code')
PARSER.add_argument('-cluster_name', type=str, required=False, default='kubeflow', help='name of the EKS cluster')
PARSER.add_argument('-iam_role_name', type=str, required=False, default='kubeflow-user-role', help='name of the IAM role attached to Kubeflow profile namespace')
PARSER.add_argument('-meth', type=str, required=True, default=None, help='method name of class KubeflowUserManagement to apply')
ARGS = PARSER.parse_args()


class KubeflowUserManagementException(Exception):
    """
    Class for handling exceptions from class KubeflowUserManagement
    """
    pass


class KubeflowUserManagement:
    """
    Class for handling Kubeflow user management
    """
    def __init__(self,
                 user_email: str,
                 user_name: str,
                 aws_account_id: str,
                 aws_region: str = 'eu-central-1',
                 cluster_name: str = 'kubeflow',
                 iam_role_name: str = 'kubeflow-user-role'
                 ):
        """
        :param user_email: str
            Name of the user

        :param user_name: str
            Name of the user

        :param aws_account_id: str
            AWS Account ID

        :param aws_region: str
            AWS region

        :param cluster_name: str
            Name of the EKS cluster

        :param iam_role_name: str
            Name of the IAM role for Kubeflow user
        """
        self.user_email: str = user_email
        self.user_name: str = user_name
        self.cluster_name: str = cluster_name
        self.aws_region: str = aws_region
        self.aws_account_id: str = aws_account_id
        self.iam_role_name: str = iam_role_name
        self._login_to_eks_cluster()

    def _adjust_user_namespace_profile_config_yaml(self) -> str:
        """
        Adjust profile config yaml file of user-namespace

        :return str
            Adjusted user-namespace profile config yaml
        """
        if self._check_profile():
            raise KubeflowUserManagementException(f'Profile namespace ({self.user_name}) already exists')
        with open('add_profile.yaml', 'r') as file:
            _profile_config_yaml = file.read()
        _profile_config_yaml = _profile_config_yaml.replace("$(PROFILE_NAME)", self.user_name)
        _profile_config_yaml = _profile_config_yaml.replace("$(PROFILE_USER)", self.user_email)
        _profile_config_yaml = _profile_config_yaml.replace("$(ACCOUNT_ID)", self.aws_account_id)
        _profile_config_yaml = _profile_config_yaml.replace("$(IAM_ROLE_NAME)", self.iam_role_name)
        return _profile_config_yaml

    def _allow_access_to_kubeflow_pipelines(self) -> str:
        """
        Enable option for allowing access to Kubeflow pipelines from internal notebook

        :return: str
             Adjusted profile pod default config yaml
        """
        with open('add_profile_pod_default.yaml', 'r') as file:
            _profile_pod_default_config_yaml = file.read()
        _profile_pod_default_config_yaml = _profile_pod_default_config_yaml.replace("$(PROFILE_NAME)", self.user_name)
        return _profile_pod_default_config_yaml

    def _check_profile(self) -> bool:
        """
        Check if profile namespace already exists

        :return: bool
            Whether a profile namespace already exists or not
        """
        _cmd: str = "kubectl get profile"
        try:
            _completed_process = subprocess.run(_cmd,
                                                shell=True,
                                                text=True,
                                                stdout=subprocess.PIPE,
                                                stderr=subprocess.PIPE
                                                )
            if _completed_process.returncode != 0:
                raise KubeflowUserManagementException("Error:", str(_completed_process.stderr))
        except Exception as e:
            raise KubeflowUserManagementException("Error:", str(e))
        if _completed_process.stdout.find(self.user_name) >= 0:
            return True
        return False

    def _login_to_eks_cluster(self) -> None:
        """
        Login to running EKS cluster
        """
        _cmd: str = f"aws eks --region {self.aws_region} update-kubeconfig --name {self.cluster_name}"
        subprocess.run(_cmd, shell=True, capture_output=False, text=True)

    def add_profile(self) -> None:
        """
        Assign Kubeflow namespace to user
        """
        _adjusted_user_namespace_profile_config_yaml: str = self._adjust_user_namespace_profile_config_yaml()
        with open('new_add_profile.yaml', 'w') as file:
            file.write(_adjusted_user_namespace_profile_config_yaml)
        subprocess.run('kubectl apply -f new_add_profile.yaml', shell=True, capture_output=False, text=True)
        _adjusted_profile_pod_default_config_yaml: str = self._allow_access_to_kubeflow_pipelines()
        with open('new_add_profile_pod_default.yaml', 'w') as file:
            file.write(_adjusted_profile_pod_default_config_yaml)
        subprocess.run('kubectl apply -f new_add_profile_pod_default.yaml', shell=True, capture_output=False, text=True)

    def delete_profile(self) -> None:
        """
        Delete Kubeflow namespace
        """
        subprocess.run(f'kubectl delete profile {self.user_name}', shell=True, capture_output=False, text=True)


if __name__ == '__main__':
    _kubeflow_user_management: KubeflowUserManagement = KubeflowUserManagement(user_email=ARGS.user_email,
                                                                               user_name=ARGS.user_name,
                                                                               aws_account_id=ARGS.aws_account_id,
                                                                               aws_region=ARGS.aws_region,
                                                                               cluster_name=ARGS.cluster_name,
                                                                               iam_role_name=ARGS.iam_role_name
                                                                               )
    if ARGS.meth == 'add_profile':
        _kubeflow_user_management.add_profile()
    elif ARGS.meth == 'delete_profile':
        _kubeflow_user_management.delete_profile()
    else:
        raise KubeflowUserManagementException(f'Method ({ARGS.meth}) not supported')
