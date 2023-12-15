import kfp
import requests

from kfp import dsl
from kfp.compiler import compiler

DEFAULT_HOST = "https://1a2da7639409e04c-dot-europe-west2.pipelines.googleusercontent.com/"


def add_container_op_parameters(container_op: dsl.component,
                                n_cpu_request: str = None,
                                n_cpu_limit: str = None,
                                n_gpu: str = None,
                                memory_request: str = '1G',
                                memory_limit: str = None,
                                ephemeral_storage_request: str = '5G',
                                ephemeral_storage_limit: str = None,
                                instance_name: str = None
                                ):
    """
    Add parameter to container operation

    :param n_cpu_request: str
        Number of requested CPU's

    :param n_cpu_limit: str
        Maximum number of requested CPU's

    :param n_gpu: str
        Maximum number of requested GPU's

    :param container_op: dsl.ContainerOp
        DSL container class

    :param memory_request: str
        Memory request

    :param memory_limit: str
        Limit of the requested memory

    :param ephemeral_storage_request: str
        Ephemeral storage request (cloud based additional memory storage)

    :param ephemeral_storage_limit: str
        Limit of the requested ephemeral storage (cloud based additional memory storage)

    :param instance_name: str
        Name of the used AWS instance (value)
    """
    if instance_name is not None:
        container_op.add_node_selector_constraint(label_name="beta.kubernetes.io/instance-type",
                                                  value=instance_name
                                                  )
    if n_cpu_request is not None:
        container_op.container.set_cpu_request(cpu=n_cpu_request)
    if n_cpu_limit is not None:
        container_op.container.set_cpu_limit(cpu=n_cpu_limit)
    if n_gpu is not None:
        container_op.container.set_gpu_limit(gpu=n_gpu, vendor='nvidia')
    container_op.container.set_memory_request(memory=memory_request)
    if memory_limit is not None:
        container_op.container.set_memory_limit(memory=memory_limit)
    container_op.container.set_ephemeral_storage_request(size=ephemeral_storage_request)
    if ephemeral_storage_limit is not None:
        container_op.container.set_ephemeral_storage_limit(size=ephemeral_storage_limit)


def get_kfp_client(host: str, namespace: str = 'kubeflow', session_cookie: str = ""):
    """
    Initialize kubeflow client

    :param host: str
        Hostname

    :param namespace: str
        Namespace
        -> kubeflow

    :param session_cookie: str
        Authentication session cookies from AWS

    :return: kfp.Client
        Initialized kubeflow client
    """
    kfp_client = kfp.Client(host=f"{host}/pipeline",
                            cookies=f"authservice_session={session_cookie}",
                            namespace=namespace)
    return kfp_client


def get_session_cookies(host: str, user_name: str, password: str):
    """
    Get session cookies from AWS for authentication purposes

    :param host: str
        Hostname

    :param user_name: str
        Username for authentication

    :param password: str
        Password for authentication

    :return: str
        Session cookie
    """
    session = requests.Session()
    response = session.get(host)
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {"login": user_name, "password": password}
    session.post(response.url, headers=headers, data=data)
    session_cookie = session.cookies.get_dict()['authservice_session']
    return session_cookie


def kfp_compile(pipeline_func, package_path: str):
    """
    Compile python code from docker container into kubeflow pipeline

    :param pipeline_func: function
        Function to compile in kubeflow pipeline

    :param package_path: str
        Package path of the local output zip file
    """
    compiler.Compiler().compile(
        pipeline_func=pipeline_func,
        package_path=package_path
    )


def kfp_upload(kfp_client,
               pipeline_func,
               package_path: str,
               pipeline_name: str,
               description: str = "test pipeline"
               ):
    """
    Upload compiled zip file to kubeflow pipeline

    :param kfp_client: kfp.Client
        Kubeflow client

    :param pipeline_func: function
        Function to compile in kubeflow pipeline

    :param package_path: str
        Package path of the local output zip file

    :param pipeline_name: str
        Name of the kubeflow pipeline

    :param description: str
        Description of the current kubeflow pipeline
    """
    kfp_compile(
        pipeline_func=pipeline_func,
        package_path=package_path
    )
    kfp_client.upload_pipeline(
        pipeline_package_path=package_path,
        pipeline_name=pipeline_name,
        description=description
    )
