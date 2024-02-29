"""

Management of Kubeflow users in dex without using external identity provider service

"""

import bcrypt
import json
import os
import random
import subprocess

from typing import Dict, List


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
                 aws_account_id: str,
                 aws_region: str = 'eu-central-1',
                 include_profile: bool = True,
                 cluster_name: str = 'kubeflow',
                 iam_role_name: str = 'kubeflow-user-role'
                 ):
        """
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
        self.user_email: str = None
        self.user_name: str = None
        self.include_profile: bool = include_profile
        self.cluster_name: str = cluster_name
        self.aws_region: str = aws_region
        self.aws_account_id: str = aws_account_id
        self.iam_role_name: str = iam_role_name

    def _add_profile(self) -> None:
        """
        Assign Kubeflow namespace to user
        """
        _adjusted_user_namespace_profile_config_yaml: str = self._adjust_user_namespace_profile_config_yaml()
        with open('new_add_profile.yaml', 'w') as file:
            file.write(_adjusted_user_namespace_profile_config_yaml)
        subprocess.run('kubectl apply -f new_add_profile.yaml', shell=True, capture_output=False, text=True)
        os.remove('new_add_profile.yaml')
        _adjusted_profile_pod_default_config_yaml: str = self._allow_access_to_kubeflow_pipelines()
        with open('new_add_profile_pod_default.yaml', 'w') as file:
            file.write(_adjusted_profile_pod_default_config_yaml)
        subprocess.run('kubectl apply -f new_add_profile_pod_default.yaml', shell=True, capture_output=False, text=True)
        subprocess.run('kubectl rollout restart deployment dex -n auth', shell=True, capture_output=False, text=True)
        os.remove('new_add_profile_pod_default.yaml')

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
        _static_password_element_counter: int = 0
        _max_static_password_elements: int = len(list(_credentials.keys()))
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
                        if _static_password_element_counter < _max_static_password_elements:
                            _key: str = list(_credentials.keys())[_static_password_element_counter]
                            if line.find(_key) >= 0:
                                _current_credential: str = line.split(_key)[1].split('\n')[0]
                                _new_line: str = line.replace(f'{_key}{_current_credential}', f'{_key} {_credentials[list(_credentials.keys())[_static_password_element_counter]]}')
                                _new_credentials.append(f'    {_new_line}')
                                _static_password_element_counter += 1
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

    def _delete_profile(self) -> None:
        """
        Delete Kubeflow namespace
        """
        subprocess.run(f'kubectl delete profile {self.user_name}', shell=True, capture_output=False, text=True)

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
    def _get_kubernetes_namespaces() -> dict:
        """
        Get Kubernetes namespaces

        :return: dict
            Kubernetes namespace information
        """
        return json.loads(subprocess.check_output(["kubectl", "get", "namespaces", "-o", "json"]))

    def add_user(self, user_email: str, user_name: str, pwd: str, hashed: bool = False) -> None:
        """
        Add new Kubeflow user including profile

        :param user_email: str
            Name of the user

        :param user_name: str
            Name of the user

        :param pwd: str
            Password

        :param hashed: bool
            Whether password in 'pwd' parameter is already hashed using bcrypt or not
        """
        self.user_email = user_email
        self.user_name = user_name
        _adjusted_dex_config_yaml: str = self._adjust_dex_config_yaml(action='add', pwd=pwd, hashed=hashed)
        with open('new_add_user_dex.yaml', 'w') as file:
            file.write(_adjusted_dex_config_yaml)
        subprocess.run('kubectl apply -f new_add_user_dex.yaml', shell=True, capture_output=False, text=True)
        os.remove('new_add_user_dex.yaml')
        if self.include_profile:
            self._add_profile()

    @staticmethod
    def check_password(pwd: str, hashed_pwd: str) -> bool:
        """
        Check whether given password match with registered password

        :param pwd: str
            un-hashed password

        :param hashed_pwd: str
            Bcrypt hashed password

        :return: bool
             Whether given password match with registered password
        """
        if bcrypt.checkpw(pwd.encode('utf-8'), hashed_pwd.encode('utf-8')):
            return True
        else:
            return False

    def delete_user(self, user_email: str, user_name: str) -> None:
        """
        Delete registered Kubeflow user including profile

        :param user_email: str
            Name of the user

        :param user_name: str
            Name of the user
        """
        self.user_email = user_email
        self.user_name = user_name
        _adjusted_dex_config_yaml: str = self._adjust_dex_config_yaml(action='delete')
        with open('new_add_user_dex.yaml', 'w') as file:
            file.write(_adjusted_dex_config_yaml)
        subprocess.run('kubectl apply -f new_add_user_dex.yaml', shell=True, capture_output=False, text=True)
        os.remove('new_add_user_dex.yaml')
        if self.include_profile:
            self._delete_profile()
        subprocess.run('kubectl rollout restart deployment dex -n auth', shell=True, capture_output=False, text=True)

    def get_kubeflow_namespaces(self) -> Dict[str, str]:
        """
        Get Kubeflow namespaces and their owners

        :return: Dict[str, str]
            Kubeflow namespaces and their owner
        """
        _kubernetes_namespaces: dict = self._get_kubernetes_namespaces()
        _namespace_owner: Dict[str, str] = {}
        for ns in _kubernetes_namespaces['items']:
            if ns['metadata'].get('annotations') is not None:
                if ns['metadata']['annotations'].get('owner') is not None:
                    _namespace_owner.update({ns['metadata']['annotations']['owner']: ns['metadata']['labels']['kubernetes.io/metadata.name']})
        return _namespace_owner

    def get_user_password(self, user_email: str) -> str:
        """
        Get user password for given user email

        :param user_email: str
            Email address

        :return: str
            Hashed password
        """
        _dex_config_yaml: str = self._get_dex_config_map_yaml()
        _next_line: bool = False
        _pwd: str = 'None'
        for line in _dex_config_yaml.split('\n'):
            if _next_line:
                _pwd = line.split('hash: ')[-1]
                _next_line = False
                break
            if line.find(user_email) > 0:
                _next_line = True
        return _pwd

    def is_registered(self, user_email: str) -> bool:
        """
        Check whether given user is already registered

        :param user_email: str
            Email address

        :return: bool
             Whether given user is already registered
        """
        _dex_config_yaml: str = self._get_dex_config_map_yaml()
        if _dex_config_yaml.find(user_email) >= 0:
            return True
        else:
            return False

    def login_to_eks_cluster(self) -> None:
        """
        Login to running EKS cluster
        """
        _cmd: str = f"aws eks --region {self.aws_region} update-kubeconfig --name {self.cluster_name}"
        subprocess.run(_cmd, shell=True, capture_output=False, text=True)

    def replace_password(self, user_email: str, new_pwd: str) -> None:
        """
        Replace existing password

        :param user_email: str
            Email address

        :param new_pwd: str
            New password
        """
        _hashed_new_pwd: str = self._generate_bcrypt_hash(pwd=new_pwd)
        _hashed_old_pwd: str = self.get_user_password(user_email=user_email)
        _dex_config_yaml: str = self._get_dex_config_map_yaml()
        _dex_config_yaml = _dex_config_yaml.replace(_hashed_old_pwd, _hashed_new_pwd)
        with open('new_add_user_dex.yaml', 'w') as file:
            file.write(_dex_config_yaml)
        subprocess.run('kubectl apply -f new_add_user_dex.yaml', shell=True, capture_output=False, text=True)
        os.remove('new_add_user_dex.yaml')

    def password_exists(self, pwd: str) -> bool:
        """
        Check whether given password already exists

        :param pwd: str
            Un-hashed password

        :return: bool
             Whether given password already exists
        """
        _dex_config_yaml: str = self._get_dex_config_map_yaml()
        _exists: bool = False
        for line in _dex_config_yaml.split('\n'):
            if line.find('hash: ') > 0:
                _pwd = line.split()[-1]
                if self.check_password(pwd=pwd, hashed_pwd=_pwd):
                    _exists = True
        return _exists

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
