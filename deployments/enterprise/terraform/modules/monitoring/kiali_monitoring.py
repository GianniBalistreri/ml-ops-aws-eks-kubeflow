"""

Setup istio service mesh monitoring using Kiali, AWS Distro OpenTelemetry (ADOT), Amazon Managed Prometheus (AMP) and Amazon Managed Grafana (AMG)

"""

import argparse
import boto3
import json
import os
import subprocess

from typing import Dict


PARSER = argparse.ArgumentParser(description="setup istio service mesh monitoring using kiali")
PARSER.add_argument('-aws_account_id', type=str, required=True, default=None, help='AWS account id')
PARSER.add_argument('-aws_region', type=str, required=False, default='eu-central-1', help='AWS region code')
PARSER.add_argument('-cluster_name', type=str, required=False, default='kubeflow', help='name of the EKS cluster')
PARSER.add_argument('-kiali_domain_name', type=str, required=False, default=None, help='complete domain name to expose kiali-server')
PARSER.add_argument('-amp_sigv4_kiali_proxy_role_name', type=str, required=False, default='amp-sigv4-proxy-query-role', help='role name name for the amp sigv4 kiali proxy iam service account')
PARSER.add_argument('-amp_sigv4_kiali_proxy_policy_name', type=str, required=False, default='amp-sigv4-kiali-proxy', help='policy name for the amp sigv4 kiali proxy iam service account')
PARSER.add_argument('-amp_workspace_name', type=str, required=False, default='kubeflow', help='name of the amp workspace')
PARSER.add_argument('-amg_workspace_name', type=str, required=False, default='kubeflow', help='name of the amg workspace')
PARSER.add_argument('-secret_name', type=str, required=False, default='kiali', help='name of the aws secret manager secret')
PARSER.add_argument('-secret_key', type=str, required=False, default='prod', help='key of the aws secret manager secret')
PARSER.add_argument('-kiali_version', type=str, required=False, default='1.59.1', help='version of the kiali deployment')
PARSER.add_argument('-meth', type=str, required=True, default=None, help='method name of class KubeflowUserManagement to apply')
ARGS = PARSER.parse_args()


class KialiException(Exception):
    """
    Class for handling exceptions from class Kiali
    """
    pass


class Kiali:
    """
    Class for handling istio service mesh monitoring
    """
    def __init__(self,
                 aws_account_id: str,
                 aws_region: str = 'eu-central-1',
                 cluster_name: str = 'kubeflow',
                 kiali_domain_name: str = None,
                 amp_sigv4_kiali_proxy_role_name: str = 'amp-sigv4-proxy-query-role',
                 amp_sigv4_kiali_proxy_policy_name: str = 'amp-sigv4-kiali-proxy',
                 amp_workspace_name: str = 'kubeflow',
                 amg_workspace_name: str = 'kubeflow',
                 secret_name: str = None,
                 secret_key: str = None,
                 kiali_version: str = '1.59.1'
                 ):
        """
        :param aws_account_id: str
            AWS Account ID

        :param aws_region: str
            AWS region

        :param cluster_name: str
            Name of the EKS cluster

        :param kiali_domain_name: str
            Complete domain name to expose Kiali-server

        :param amp_sigv4_kiali_proxy_role_name: str
            Name of the IAM role for the AMP Sigv4 Kiali proxy IAM service account

        :param amp_sigv4_kiali_proxy_policy_name: str
            Name of the IAM policy for the AMP Sigv4 Kiali proxy IAM service account

        :param amp_workspace_name: str
            Name of the Amazon Managed Prometheus workspace

        :param amg_workspace_name: str
            Name of the Amazon Managed Grafana workspace

        :param secret_name: str
            Name of the AWS secret manager secret

        :param secret_key: str
            Key of the AWS secret manager secret

        :param kiali_version: str
            Kiali version
        """
        self.cluster_name: str = cluster_name
        self.aws_region: str = aws_region
        self.aws_account_id: str = aws_account_id
        self.kiali_domain_name: str = '""' if kiali_domain_name is None else kiali_domain_name
        self.kiali_version: str = kiali_version
        self.amp_workspace_name: str = amp_workspace_name
        self.amg_workspace_name: str = amg_workspace_name
        self.amg_workspace_id: str = None
        self.secret_name: str = secret_name
        self.secret_key: str = secret_key
        self.amp_sigv4_kiali_proxy_role_name: str = amp_sigv4_kiali_proxy_role_name
        self.amp_sigv4_kiali_proxy_policy_name: str = amp_sigv4_kiali_proxy_policy_name
        subprocess.run(f"aws eks --region {self.aws_region} update-kubeconfig --name {self.cluster_name}", shell=True)

    def _adjust_otel_collector_xray_prometheus_complete_config_yaml(self) -> str:
        """
        Adjust OpenTelemetry Collector config yaml file

        :return: str
            Adjusted OpenTelemetry Collector config yaml
        """
        _amp_workspace_id: str = self._get_amp_workspace_id()
        with open('otel_collector_xray_prometheus_complete.yaml', 'r') as file:
            _otel_collector_config_yaml = file.read()
        _otel_collector_config_yaml = _otel_collector_config_yaml.replace("$(AWS_REGION)", self.aws_region)
        _otel_collector_config_yaml = _otel_collector_config_yaml.replace("$(AMP_WORKSPACE_ID)", _amp_workspace_id)
        return _otel_collector_config_yaml

    def _adjust_kiali_config_yaml(self) -> str:
        """
        Adjust Kiali config yaml file

        :return str
            Adjusted Kiali config yaml
        """
        with open('kiali.yaml', 'r') as file:
            _kiali_config_yaml = file.read()
        _amp_workspace_id: str = self._get_amp_workspace_id()
        #_amg_workspace_info: Dict[str, str] = self._get_amg_workspace_info()
        #self.amg_workspace_id = _amg_workspace_info.get('id')
        #_amg_api_key: str = self._get_amg_api_key()
        #self._update_secret_manager_secret(amg_api_key=_amg_api_key)
        _kiali_config_yaml = _kiali_config_yaml.replace("$(KIALI_VERSION)", self.kiali_version)
        _kiali_config_yaml = _kiali_config_yaml.replace("$(AMP_WORKSPACE_ID)", _amp_workspace_id)
        #_kiali_config_yaml = _kiali_config_yaml.replace("$(AMG_API_KEY)", _amg_api_key)
        #_kiali_config_yaml = _kiali_config_yaml.replace("$(AMG_WORKSPACE_URL)", _amg_workspace_info.get('url'))
        _kiali_config_yaml = _kiali_config_yaml.replace("$(KIALI_DOMAIN_URL)", self.kiali_domain_name)
        return _kiali_config_yaml

    def _adjust_kiali_sigv4_config_yaml(self) -> str:
        """
        Adjust Kiali SigV4 proxy config yaml file

        :return str
            Adjusted Kiali SigV4 proxy config yaml
        """
        with open('kiali_sigv4.yaml', 'r') as file:
            _kiali_sigv4_config_yaml = file.read()
        _kiali_sigv4_config_yaml = _kiali_sigv4_config_yaml.replace("$(AMP_SIGV4_IAM_ROLE)", self.amp_sigv4_kiali_proxy_role_name)
        _kiali_sigv4_config_yaml = _kiali_sigv4_config_yaml.replace("$(AWS_REGION)", self.aws_region)
        return _kiali_sigv4_config_yaml

    def _associate_iam_oidc_provider(self) -> None:
        """
        Associate IAM OIDC provider
        """
        subprocess.run(f'eksctl utils associate-iam-oidc-provider --cluster={self.cluster_name} --approve', shell=True)

    def _create_adot_addon(self) -> None:
        """
        Create AWS Distro OpenTelemetry EKS addon
        """
        _cmd_get_iam_serviceaccount: str = f"eksctl get iamserviceaccount --cluster={self.cluster_name} --namespace=opentelemetry-operator-system"
        _output = subprocess.run(_cmd_get_iam_serviceaccount, shell=True, capture_output=True, text=True)
        _role_arn: str = _output.stdout.splitlines()[1].split()[-1]
        #_cmd_create_addon: str = f"aws eks create-addon --addon-name adot --cluster-name {self.cluster_name} --service-account-role-arn {_role_arn}"
        subprocess.run(f"aws eks create-addon --addon-name adot --cluster-name {self.cluster_name}", shell=True)

    def _create_cloudwatch_observability_addon(self) -> None:
        """
        Create Amazon CloudWatch Observability addon
        """
        subprocess.run(f"aws eks create-addon --addon-name amazon-cloudwatch-observability --cluster-name {self.cluster_name} --region {self.aws_region}", shell=True)

    def _create_iam_service_account_amp_sigv4_kiali_proxy(self) -> None:
        """
        Create eks iam service account for AMP Sigv4 Kiali proxy
        """
        _cmd: str = f'eksctl create iamserviceaccount \
        --name {self.amp_sigv4_kiali_proxy_role_name} \
        --namespace istio-system \
        --cluster {self.cluster_name} \
        --region {self.aws_region} \
        --attach-policy-arn arn:aws:iam::{self.aws_account_id}:policy/{self.amp_sigv4_kiali_proxy_policy_name} \
        --approve \
        --override-existing-serviceaccounts'
        subprocess.run(_cmd, shell=True)

    def _create_iam_service_account_aws_otel_eks(self) -> None:
        """
        Create eks iam service account for AWS OpenTelemetry EKS addon
        """
        _cmd: str = f'eksctl create iamserviceaccount \
        --name aws-otel-collector \
        --namespace aws-otel-eks \
        --cluster {self.cluster_name} \
        --region {self.aws_region} \
        --role-name EKS-ADOT-ServiceAccount-Role \
        --attach-policy-arn arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy \
        --attach-policy-arn arn:aws:iam::aws:policy/AmazonPrometheusRemoteWriteAccess \
        --attach-policy-arn arn:aws:iam::aws:policy/AWSXRayDaemonWriteAccess \
        --approve \
        --override-existing-serviceaccounts'
        subprocess.run(_cmd, shell=True)

    def _deploy_adot(self) -> None:
        """
        Deploy AWS Distro OpenTelemetry addon
        """
        subprocess.run('kubectl apply -f addon_otel_permissions.yaml', shell=True)
        self._create_adot_addon()
        self._associate_iam_oidc_provider()
        self._create_iam_service_account_aws_otel_eks()
        _adjusted_otel_collector_config_yaml: str = self._adjust_otel_collector_xray_prometheus_complete_config_yaml()
        with open('new_otel_collector_xray_prometheus_complete.yaml', 'w') as file:
            file.write(_adjusted_otel_collector_config_yaml)
        subprocess.run('kubectl apply -f new_otel_collector_xray_prometheus_complete.yaml', shell=True)
        #os.remove('new_otel_collector_xray_prometheus_complete.yaml')

    def _get_amg_api_key(self) -> str:
        """
        Get AWS Managed Grafana (AMG) API key

        :return: str
            AMG API key
        """
        _cmd_get_amg_api_key: str = f"aws grafana create-workspace-api-key --key-name 'kiali_access_key' --key-role 'ADMIN' --seconds-to-live 432000 --workspace-id '{self.amg_workspace_id}' | jq '.key'"
        return subprocess.run(_cmd_get_amg_api_key, shell=True, capture_output=True, text=True).stdout.split('\n')[0][1:-1]

    def _get_amg_workspace_info(self) -> Dict[str, str]:
        """
        Get AWS Managed Grafana (AMG) workspace information

        :return: Dict[str, str]
            AMG workspace id and url
        """
        _amg_workspace_info: Dict[str, str] = dict(id='', url='')
        _client: boto3 = boto3.client('grafana', region_name=self.aws_region)
        _response: dict = _client.list_workspaces()
        _found_workspace: bool = False
        for workspace in _response['workspaces']:
            if workspace.get('name') == self.amg_workspace_name:
                _found_workspace = True
                _amg_workspace_info.update({'id': workspace.get('id'), 'url': workspace.get('endpoint')})
                break
        if not _found_workspace:
            raise KialiException(f'Given AWS Managed Grafana workspace name ({self.amg_workspace_name}) not found')
        return _amg_workspace_info

    def _get_amp_workspace_id(self) -> str:
        """
        Get AWS Managed Prometheus (AMP) workspace id

        :return: str
            AMP workspace id
        """
        _cmd_get_amp_workspace_id: str = f"aws amp list-workspaces --alias {self.amp_workspace_name} | jq -r '.workspaces[].workspaceId'"
        return subprocess.run(_cmd_get_amp_workspace_id, shell=True, capture_output=True, text=True).stdout.split('\n')[0]

    def _update_secret_manager_secret(self, kiali_token: str = None, amg_api_key: str = None) -> None:
        """
        Update AWS secret manager secret

        :param kiali_token: str
            Generated Kiali login token

        :param amg_api_key: str
            Generated Amazon Managed Grafana API key
        """
        _client: boto3 = boto3.client('secretsmanager')
        try:
            _response: dict = _client.describe_secret(SecretId=self.secret_name)
            _create_kiali_secret_name: bool = False
        except _client.exceptions.ResourceNotFoundException:
            _create_kiali_secret_name: bool = True
        _secret_key: str = self.amg_workspace_id if kiali_token is None else self.secret_key
        _secret_value: str = amg_api_key if kiali_token is None else kiali_token
        if _create_kiali_secret_name:
            _secret_string: dict = dict(_secret_key=_secret_value)
            _response: dict = _client.create_secret(Name=self.secret_name,
                                                    Description='Kiali secrets',
                                                    SecretString=str(_secret_string)
                                                    )
            if _response['ResponseMetadata']['HTTPStatusCode'] != 200:
                raise KialiException(f'Secret (name={self.secret_name}, key={_secret_key}) could not be updated (status_code={_response["ResponseMetadata"]["HTTPStatusCode"]})')
        else:
            _response: dict = _client.get_secret_value(SecretId=self.secret_name)
            _secret_values: dict = json.loads(_response['SecretString'])
            _secret_values[_secret_key] = _secret_value
            _new_secret_value: json = json.dumps(_secret_values)
            _response: dict = _client.update_secret(SecretId=self.secret_name, SecretString=_new_secret_value)
            if _response['ResponseMetadata']['HTTPStatusCode'] != 200:
                raise KialiException(f'Secret (name={self.secret_name}, key={_secret_key}) could not be updated (status_code={_response["ResponseMetadata"]["HTTPStatusCode"]})')

    def deploy(self) -> None:
        """
        Deploy Kiali as well as periphery services
        """
        self._deploy_adot()
        self._create_cloudwatch_observability_addon()
        self._create_iam_service_account_amp_sigv4_kiali_proxy()
        _adjusted_kiali_sigv4_config_yaml: str = self._adjust_kiali_sigv4_config_yaml()
        with open('new_kiali_sigv4.yaml', 'w') as file:
            file.write(_adjusted_kiali_sigv4_config_yaml)
        subprocess.run('kubectl apply -f new_kiali_sigv4.yaml', shell=True)
        os.remove('new_kiali_sigv4.yaml')
        _adjusted_kiali_config_yaml: str = self._adjust_kiali_config_yaml()
        with open('new_kiali.yaml', 'w') as file:
            file.write(_adjusted_kiali_config_yaml)
        subprocess.run('kubectl apply -f new_kiali.yaml', shell=True)
        #os.remove('new_kiali.yaml')
        #_kiali_token: str = subprocess.run('kubectl -n istio-system create token kiali', shell=True, capture_output=True, text=True).stdout
        #self._update_secret_manager_secret(kiali_token=_kiali_token)

    def delete(self) -> None:
        """
        Delete Kiali deployment as well as periphery services
        """
        subprocess.run('kubectl delete deployment kiali-sigv4 -n istio-system', shell=True)
        subprocess.run('kubectl delete service kiali-sigv4 -n istio-system', shell=True)
        subprocess.run('kubectl delete deployment kiali -n istio-system', shell=True)
        subprocess.run('kubectl delete service kiali -n istio-system', shell=True)
        _cmd_delete_iam_service_account: str = f'eksctl delete iamserviceaccount --name {self.amp_sigv4_kiali_proxy_role_name} --namespace istio-system --cluster {self.cluster_name}'
        subprocess.run(_cmd_delete_iam_service_account, shell=True)
        _cmd_delete_iam_service_account: str = f'eksctl delete iamserviceaccount --name adot-collector --namespace aws-otel-eks --cluster {self.cluster_name}'
        subprocess.run(_cmd_delete_iam_service_account, shell=True)
        subprocess.run(f'eksctl delete addon --cluster={self.cluster_name} --name=adot', shell=True)
        subprocess.run(f'eksctl delete addon --cluster={self.cluster_name} --name=amazon-cloudwatch-observability', shell=True)


if __name__ == '__main__':
    _kiali: Kiali = Kiali(aws_account_id=ARGS.aws_account_id,
                          aws_region=ARGS.aws_region,
                          cluster_name=ARGS.cluster_name,
                          kiali_domain_name=ARGS.kiali_domain_name,
                          amp_sigv4_kiali_proxy_role_name=ARGS.amp_sigv4_kiali_proxy_role_name,
                          amp_sigv4_kiali_proxy_policy_name=ARGS.amp_sigv4_kiali_proxy_policy_name,
                          amp_workspace_name=ARGS.amp_workspace_name,
                          amg_workspace_name=ARGS.amg_workspace_name,
                          secret_name=ARGS.secret_name,
                          secret_key=ARGS.secret_key,
                          kiali_version=ARGS.kiali_version
                          )
    if ARGS.meth == 'deploy':
        _kiali.deploy()
    elif ARGS.meth == 'delete':
        _kiali.delete()
    else:
        raise KialiException(f'Method ({ARGS.meth}) not supported')
