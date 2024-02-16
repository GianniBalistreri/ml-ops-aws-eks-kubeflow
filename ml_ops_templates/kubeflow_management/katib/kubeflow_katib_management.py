"""

Management of katib setup

"""

import argparse
import subprocess

from typing import List


PARSER = argparse.ArgumentParser(description="manage kubeflow hyperparameter tuning service called katib")
PARSER.add_argument('-aws_account_id', type=str, required=True, default=None, help='AWS account id')
PARSER.add_argument('-service_account_name', type=str, required=False, default='sa', help='name of the service account')
PARSER.add_argument('-aws_region', type=str, required=False, default='eu-central-1', help='AWS region code')
PARSER.add_argument('-cluster_name', type=str, required=False, default='kubeflow', help='name of the EKS cluster')
ARGS = PARSER.parse_args()


class KubeflowKatibManagementException(Exception):
    """
    Class for handling exceptions from class KubeflowKatibManagement
    """
    pass


class KubeflowKatibManagement:
    """
    Class for handling Kubeflow katib management
    """
    def __init__(self,
                 aws_account_id: str,
                 service_account_name: str,
                 aws_region: str = 'eu-central-1',
                 cluster_name: str = 'kubeflow'
                 ):
        """
        :param aws_account_id: str
            AWS Account ID

        :param service_account_name: str
            Name of the service account name

        :param aws_region: str
            AWS region

        :param cluster_name: str
            Name of the EKS cluster
        """
        self.service_account_name: str = service_account_name
        self.cluster_name: str = cluster_name
        self.aws_region: str = aws_region
        self.aws_account_id: str = aws_account_id
        self._login_to_eks_cluster()

    def _adjust_katib_config(self) -> str:
        """
        Adjust Katib config yaml

        :return: str
            Adjusted katib config yaml file
        """
        _cmd: str = f"kubectl get configMap katib-config -n {self.cluster_name} -o yaml > katib_config.yaml"
        subprocess.run(_cmd, shell=True, capture_output=False, text=True)
        with open('katib_config.yaml', 'r') as file:
            _katib_config_yaml = file.read()
        _config_yaml: List[str] = []
        _found_area: bool = False
        for line in _katib_config_yaml.split('\n'):
            if line.find('suggestion: |-') >= 0:
                _found_area = True
            if line.find('kind: ConfigMap') >= 0:
                _found_area = False
            _config_yaml.append(line)
            if _found_area:
                if line.find('"image":') >= 0:
                    _config_yaml.append(f'        "serviceAccountName": "{self.service_account_name}"')
        return "\n".join(_config_yaml)

    def _login_to_eks_cluster(self) -> None:
        """
        Login to running EKS cluster
        """
        _cmd: str = f"aws eks --region {self.aws_region} update-kubeconfig --name {self.cluster_name}"
        subprocess.run(_cmd, shell=True, capture_output=False, text=True)

    def enable_katib(self) -> None:
        """
        Enable katib by accessing S3 AWS services
        """
        _katib_config_yaml: str = self._adjust_katib_config()
        with open('new_katib_config.yaml', 'w') as file:
            file.write(_katib_config_yaml)
        subprocess.run('kubectl apply -f new_katib_config.yaml', shell=True, capture_output=False, text=True)


if __name__ == '__main__':
    _kubeflow_katib_management: KubeflowKatibManagement = KubeflowKatibManagement(aws_account_id=ARGS.aws_account_id,
                                                                                  service_account_name=ARGS.service_account_name,
                                                                                  aws_region=ARGS.aws_region,
                                                                                  cluster_name=ARGS.cluster_name
                                                                                  )
    _kubeflow_katib_management.enable_katib()
