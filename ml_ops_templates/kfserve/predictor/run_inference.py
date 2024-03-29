import requests
import os
import json


def run_inference_sample():
    # common vars
    KUBEFLOW_DOMAIN = os.environ.get("KUBEFLOW_DOMAIN", "dev.shopware-kubeflow.net")
    PROFILE_NAMESPACE = os.environ.get("PROFILE_NAMESPACE", "production")
    MODEL_NAME = os.environ.get("MODEL_NAME", "occupancy-xgb-replicated")
    AUTH_PROVIDER = os.environ.get("AUTH_PROVIDER", "dex")

    URL = f"https://{MODEL_NAME}.{PROFILE_NAMESPACE}.{KUBEFLOW_DOMAIN}/v1/models/{MODEL_NAME}:predict"
    HEADERS = {"Host": f"{MODEL_NAME}.{PROFILE_NAMESPACE}.{KUBEFLOW_DOMAIN}"}
    DASHBOARD_URL = f"https://kubeflow.{KUBEFLOW_DOMAIN}"
    with open('request_data_occupancy.json', 'r') as _file:
        data = json.load(_file)
    #data = load_json_file("./utils/kserve/iris-input.json")
    response = None
    if AUTH_PROVIDER != "cognito":
        PROFILE_USERNAME = os.environ.get("PROFILE_USERNAME", "g.balistreri@shopware.com")
        PASSWORD = os.environ.get("PASSWORD", "Ich_Rocke_Arcana_Analytica_762")

        def session_cookie(host, login, password):
            session = requests.Session()
            response = session.get(host)
            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
            }
            data = {"login": login, "password": password}
            session.post(response.url, headers=headers, data=data)
            session_cookie = session.cookies.get_dict()["authservice_session"]
            return session_cookie

        cookie = {
            "authservice_session": session_cookie(
                DASHBOARD_URL, PROFILE_USERNAME, PASSWORD
            )
        }
        response = requests.post(URL, headers=HEADERS, json=data, cookies=cookie)
    else:
        HTTP_HEADER_NAME = os.environ.get("HTTP_HEADER_NAME", "x-api-key")
        HTTP_HEADER_VALUE = os.environ.get("HTTP_HEADER_VALUE", "token1")
        HEADERS[HTTP_HEADER_NAME] = HTTP_HEADER_VALUE

        response = requests.post(URL, headers=HEADERS, json=data)

    status_code = response.status_code
    #print(response.json())
    print("Status Code", status_code)
    if status_code == 200:
        print("JSON Response ", json.dumps(response.json(), indent=2))

    else:
        raise Exception("prediction failed, status code = ")


if __name__ == "__main__":
    for i in range(0, 1000, 1):
        run_inference_sample()
