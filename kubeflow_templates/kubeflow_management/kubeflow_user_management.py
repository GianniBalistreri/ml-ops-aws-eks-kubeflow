"""
Management of Kubeflow users in dex without using external identity provider service
"""

import argparse
import bcrypt
import random
import re
import subprocess

from typing import List


PARSER = argparse.ArgumentParser(description="manage kubeflow user")
PARSER.add_argument('-user_email', type=str, required=True, default=None, help='user email address')
PARSER.add_argument('-user_name', type=str, required=True, default=None, help='user name')
PARSER.add_argument('-aws_account_id', type=str, required=True, default=None, help='AWS account id')
PARSER.add_argument('-aws_region', type=str, required=False, default='eu-central-1', help='AWS region code')
PARSER.add_argument('-include_profile', type=str, required=False, default=True, help='whether to create Kubeflow profile namespace or not (multi tenancy)')
PARSER.add_argument('-cluster_name', type=str, required=False, default='kubeflow', help='name of the EKS cluster')
PARSER.add_argument('-iam_role_name', type=str, required=False, default='kubeflow-user-role', help='name of the IAM role attached to Kubeflow profile namespace')
PARSER.add_argument('-meth', type=str, required=True, default=None, help='method name of class KubeflowUserManagement to apply')
PARSER.add_argument('-pwd', type=str, required=True, default=None, help='user password')
PARSER.add_argument('-hashed', type=bool, required=False, default=False, help='whether user password is already hashed by bcrypt or not')
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
                 include_profile: bool = True,
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

        :param include_profile: bool
            Whether to include profile adjustments automatically or not

        :param cluster_name: str
            Name of the EKS cluster

        :param iam_role_name: str
            Name of the IAM role for Kubeflow user
        """
        self.user_email: str = user_email
        self.user_name: str = user_name
        self.include_profile: bool = include_profile
        self.cluster_name: str = cluster_name
        self.aws_region: str = aws_region
        self.aws_account_id: str = aws_account_id
        self.iam_role_name: str = iam_role_name
        self._login_to_eks_cluster()

    def _adjust_dex_config_yaml(self, action: str, pwd: str = None, hashed: bool = True) -> str:
        """
        Adjust dex config yaml file

        :param action
            Name of the action to take
                -> add: Add new user
                -> delete: Delete user

        :param pwd: str
            Password

        :param hashed: bool
            Whether password in 'pwd' parameter is already hashed using bcrypt or not

        :return str
            Adjusted dex config yaml file
        """
        _pwd: str = pwd if hashed else self._generate_bcrypt_hash(pwd=pwd)
        _credentials: dict = {'- email:': self.user_email,
                              'hash:': _pwd,
                              'username:': self.user_name,
                              'userID:': f'"{self._generate_random_user_id()}"'
                              }
        _dex_config_yaml: str = self._get_dex_config_map_yaml()
        if action == 'add' and _dex_config_yaml.find(f'- email: {self.user_email}') >= 0:
            raise KubeflowUserManagementException(f'User email address ({self.user_email}) already exists')
        if action == 'delete' and _dex_config_yaml.find(f'- email: {self.user_email}') < 0:
            raise KubeflowUserManagementException(f'User email address ({self.user_email}) not found')
        _found_target_to_delete: bool = False
        _found_credential_section: bool = False
        _counter: int = 0
        _new_credentials: List[str] = []
        _config_yaml: List[str] = ['apiVersion: v1', 'data:', '  config.yaml: |']
        for line in _dex_config_yaml.split('\n'):
            if _found_credential_section:
                if line.find('staticClients:') >= 0:
                    _found_credential_section = False
                    for new_cred in _new_credentials:
                        _config_yaml.append(new_cred)
                else:
                    if action == 'add':
                        _key: str = list(_credentials.keys())[_counter]
                        if line.find(_key) >= 0:
                            _current_credential: str = line.split(_key)[1].split('\n')[0]
                            _new_line: str = line.replace(f'{_key}{_current_credential}', f'{_key} {_credentials[list(_credentials.keys())[_counter]]}')
                            _new_credentials.append(f'    {_new_line}')
                            _counter += 1
            else:
                if line.find('staticPasswords:') >= 0:
                    _found_credential_section = True
            if action == 'delete' and line.find(self.user_email) >= 0:
                _found_target_to_delete = True
                continue
            if _found_target_to_delete:
                if line.find('staticClients:') < 0 and line.find('- email:') < 0:
                    continue
                else:
                    _found_target_to_delete = False
            _config_yaml.append(f'    {line}')
            if line.find('secretEnv') > 0:
                break
        _config_yaml.append('kind: ConfigMap')
        _config_yaml.append('metadata:')
        _config_yaml.append('  name: dex')
        _config_yaml.append('  namespace: auth')
        return "\n".join(_config_yaml)

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

    @staticmethod
    def _generate_bcrypt_hash(pwd: str) -> str:
        """
        Generate hash for password using bcrypt (hashing method used in dex)

        :param pwd: str
            Original password

        :return: str
            Hashed password
        """
        _salted_hash: bytes = bcrypt.hashpw(pwd.encode('utf-8'), bcrypt.gensalt())
        return bcrypt.hashpw(pwd.encode('utf-8'), _salted_hash).decode()

    @staticmethod
    def _generate_random_user_id() -> str:
        """
        Generate random user id (14 digits)

        :return: str
            Randomly generate user id
        """
        return ''.join([str(random.randint(0, 9)) for _ in range(0, 14, 1)])

    @staticmethod
    def _get_dex_config_map_yaml() -> str:
        """
        Get deployed dex_config_yaml file

        :return: str
            File content
        """
        _cmd: str = "kubectl get configmap dex -n auth -o jsonpath='{.data.config\.yaml}'"
        try:
            _completed_process = subprocess.run(_cmd,
                                                shell=True,
                                                text=True,
                                                stdout=subprocess.PIPE,
                                                stderr=subprocess.PIPE
                                                )
            if _completed_process.returncode == 0:
                return _completed_process.stdout
            else:
                raise KubeflowUserManagementException("Error:", str(_completed_process.stderr))
        except Exception as e:
            raise KubeflowUserManagementException("Error:", str(e))

    @staticmethod
    def _check_password(pwd: str) -> None:
        """
        Check whether given password is valid based on several condition or not

        :param pwd: str
            Password
        """
        if len(pwd) < 10:
            raise KubeflowUserManagementException('Password must have at least 10 characters')
        if not re.search(r'\d', pwd):
            raise KubeflowUserManagementException('Password must contain numbers')
        if not re.search(r'[!@#$%^&*()_+{}\[\]:;<>,.?~\\-]', pwd):
            raise KubeflowUserManagementException('Password must contain special characters')
        if not re.search(r'[A-Z]', pwd):
            raise KubeflowUserManagementException('Password must contain capital characters')

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
        subprocess.run('kubectl rollout restart deployment dex -n auth', shell=True, capture_output=False, text=True)

    def add_user(self, pwd: str, hashed: bool = False) -> None:
        """
        Add new Kubeflow user including profile

        :param pwd: str
            Password

        :param hashed: bool
            Whether password in 'pwd' parameter is already hashed using bcrypt or not
        """
        self._check_password(pwd=pwd)
        _adjusted_dex_config_yaml: str = self._adjust_dex_config_yaml(action='add', pwd=pwd, hashed=hashed)
        with open('new_add_user_dex.yaml', 'w') as file:
            file.write(_adjusted_dex_config_yaml)
        subprocess.run('kubectl apply -f new_add_user_dex.yaml', shell=True, capture_output=False, text=True)
        if self.include_profile:
            self.add_profile()

    def delete_profile(self) -> None:
        """
        Delete Kubeflow namespace
        """
        subprocess.run(f'kubectl delete profile {self.user_name}', shell=True, capture_output=False, text=True)

    def delete_user(self) -> None:
        """
        Delete registered Kubeflow user including profile
        """
        _adjusted_dex_config_yaml: str = self._adjust_dex_config_yaml(action='delete')
        with open('new_add_user_dex.yaml', 'w') as file:
            file.write(_adjusted_dex_config_yaml)
        subprocess.run('kubectl apply -f new_add_user_dex.yaml', shell=True, capture_output=False, text=True)
        if self.include_profile:
            self.delete_profile()
        subprocess.run('kubectl rollout restart deployment dex -n auth', shell=True, capture_output=False, text=True)

    def initial_deployment(self, pwd: str, hashed: bool = False) -> None:
        """
        Add new user and delete default user settings for initial deployment

        :param pwd: str
            Password

        :param hashed: bool
            Whether password in 'pwd' parameter is already hashed using bcrypt or not
        """
        self.add_user(pwd=pwd, hashed=hashed)
        self.user_email = 'user@example.com'
        self.user_name = 'kubeflow-user-example-com'
        self.delete_user()


if __name__ == '__main__':
    _kubeflow_user_management: KubeflowUserManagement = KubeflowUserManagement(user_email=ARGS.user_email,
                                                                               user_name=ARGS.user_name,
                                                                               aws_account_id=ARGS.aws_account_id,
                                                                               aws_region=ARGS.aws_region,
                                                                               include_profile=ARGS.include_profile,
                                                                               cluster_name=ARGS.cluster_name,
                                                                               iam_role_name=ARGS.iam_role_name
                                                                               )
    if ARGS.meth == 'add_profile':
        _kubeflow_user_management.add_profile()
    elif ARGS.meth == 'add_user':
        _kubeflow_user_management.add_user(pwd=ARGS.pwd, hashed=ARGS.hashed)
    elif ARGS.meth == 'delete_profile':
        _kubeflow_user_management.delete_profile()
    elif ARGS.meth == 'delete_user':
        _kubeflow_user_management.delete_user()
    elif ARGS.meth == 'initial_deployment':
        _kubeflow_user_management.initial_deployment(pwd=ARGS.pwd, hashed=ARGS.hashed)
    else:
        raise KubeflowUserManagementException(f'Method ({ARGS.meth}) not supported')
