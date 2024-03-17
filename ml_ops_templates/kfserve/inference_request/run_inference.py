"""

Inference request for KServe model deployment

"""

import json
import os
import requests


class KServeInferenceException(Exception):
    """
    Class for handling exceptions for class KServeInference
    """
    pass


class KServeInference:
    """
    Class for preparing and sending request to deployed KServe model endpoint
    """
    def __init__(self,
                 model_name: str,
                 profile_namespace: str,
                 top_level_domain_name: str,
                 second_level_domain_name: str,
                 subdomain_name: str,
                 data_file_name: str,
                 user_name: str,
                 pwd: str,
                 auth_provider: str = 'dex',
                 ):
        """
        :param model_name: str
            Name of the deployed model

        :param profile_namespace: str
            Name of the profile namespace

        :param top_level_domain_name: str
            Name of the top level domain

        :param second_level_domain_name: str
            Name of the second level domain

        :param subdomain_name: str
            Name of the subdomain

        :param data_file_name: str
            Complete file path to the request data (json format)

        :param user_name: str
            Name of the Kubeflow user

        :param pwd: str
            Kubeflow user password

        :param auth_provider: str
            Name of the authentication provider
                -> dex: Dex
                -> cognito: AWS Cognito
        """
        self.model_name: str = model_name
        self.profile_namespace: str = profile_namespace
        self.data_file_name: str = data_file_name
        self.user_name: str = user_name
        self.pwd: str = pwd
        self.domain: str = f'{subdomain_name}.{second_level_domain_name}.{top_level_domain_name}'
        self.host: str = f'https://kubeflow.{self.domain}'
        self.headers: dict = {'Host': f'{model_name}.{profile_namespace}.{self.domain}'}
        self.url: str = f'https://{model_name}.{profile_namespace}.{self.domain}/v1/models/{model_name}:predict'
        self.auth_provider: str = auth_provider

    def _get_session_cookie(self) -> str:
        """
        Get dex session cookie

        :return: str
            Session cookie
        """
        _session: requests.Session = requests.Session()
        _response: requests.Response = _session.get(self.host)
        _headers: dict = {'Content-Type': 'application/x-www-form-urlencoded'}
        _data: dict = {'login': self.user_name, 'password': self.pwd}
        _session.post(_response.url, headers=_headers, data=_data)
        return _session.cookies.get_dict()['authservice_session']

    def main(self) -> list:
        """
        Send request to deployed KServe model endpoint and receive predictions

        :return: list
            Predictions
        """
        with open(self.data_file_name, 'r') as _file:
            data = json.load(_file)
        if self.auth_provider == 'dex':
            _cookie: dict = {'authservice_session': self._get_session_cookie()}
            _response: requests.Response = requests.post(self.url, headers=self.headers, json=data, cookies=_cookie)
        elif self.auth_provider == 'cognito':
            _http_header_name: str = 'x-api-key'
            _http_header_value: str = 'token1'
            self.headers[_http_header_name] = _http_header_value
            _response: requests.Response = requests.post(self.url, headers=self.headers, json=data)
        else:
            raise KServeInferenceException(f'Authentication provider ({self.auth_provider}) not supported')
        status_code = _response.status_code
        if status_code == 200:
            return _response.json().get('predictions')
        else:
            raise KServeInferenceException(f"Prediction failed (status code = {status_code})")


if __name__ == '__main__':
    _kserve_inference: KServeInference = KServeInference(model_name=os.environ.get('MODEL_NAME'),
                                                         profile_namespace=os.environ.get('PROFILE_NAMESPACE'),
                                                         top_level_domain_name=os.environ.get('TOP_LEVEL_DOMAIN_NAME'),
                                                         second_level_domain_name=os.environ.get('SECOND_LEVEL_DOMAIN_NAME'),
                                                         subdomain_name=os.environ.get('SUBDOMAIN_NAME'),
                                                         data_file_name=os.environ.get('DATA_FILE_NAME'),
                                                         user_name=os.environ.get('USER_NAME'),
                                                         pwd=os.environ.get('PWD'),
                                                         auth_provider=os.environ.get('AUTH_PROVIDER', 'dex')
                                                         )
    _prediction: list = _kserve_inference.main()
    print(_prediction)
