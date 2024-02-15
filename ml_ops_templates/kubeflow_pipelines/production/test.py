import os
import kfp

from kubeflow_pipelines.production.v1.get_session_cookie_dex import get_istio_auth_session
#from kfp.client import Client

_auth: dict = get_istio_auth_session(url=os.getenv('KF_URL'),
                                     username=os.getenv('USER_NAME'),
                                     password=os.getenv('PWD')
                                     )
print(_auth)
_auth_cookies: str = _auth["session_cookie"]
print(_auth_cookies)
kfp_client = kfp.Client(host=f'{os.getenv("KF_URL")}/pipeline',
                    client_id=None,
                    namespace='production',
                    other_client_id=None,
                    other_client_secret=None,
                    existing_token=None,
                    cookies=_auth_cookies,
                    proxy=None,
                    ssl_ca_cert=None,
                    kube_context=None,
                    credentials=None,
                    ui_host=None,
                    )

print(kfp_client.get_user_namespace())
print(kfp_client.list_pipelines())
print(kfp_client.list_runs())
print(kfp_client.list_experiments())
