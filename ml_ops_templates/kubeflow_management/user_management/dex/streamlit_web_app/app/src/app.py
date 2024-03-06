"""

Streamlit web application for managing kubeflow user using dex as authentication provider

"""

import os
import re
import streamlit as st

from custom_logger import Log
from kubeflow_user_management import KubeflowUserManagement
from typing import Dict


KUBEFLOW_USER_MANAGEMENT: KubeflowUserManagement = KubeflowUserManagement(aws_account_id=os.getenv('AWS_ACCOUNT_ID'),
                                                                          aws_region=os.getenv('AWS_REGION'),
                                                                          include_profile=True,
                                                                          cluster_name=os.getenv('CLUSTER_NAME'),
                                                                          iam_role_name='kubeflow-user-role'
                                                                          )


def _check_credentials(email: str, namespace: str, pwd: str) -> bool:
    """
    Check whether credentials are match or not

    :param email: str
        Email address

    :param namespace: str
        Namespace

    :param pwd: str
        Password

    :return: bool
        Whether credentials are match or not
    """
    _kubeflow_namespace: Dict[str, str] = KUBEFLOW_USER_MANAGEMENT.get_kubeflow_namespaces()
    if _kubeflow_namespace.get(email) is None:
        st.error(f"Email ({email}) not registered")
        Log().log(msg=f'Email ({email}) not registered')
        return False
    if _kubeflow_namespace.get(email) == namespace:
        _hashed_pwd: str = KUBEFLOW_USER_MANAGEMENT.get_user_password(user_email=email)
        return KUBEFLOW_USER_MANAGEMENT.check_password(pwd=pwd, hashed_pwd=_hashed_pwd)
    else:
        st.error(f"Namespace ({namespace}) is not assigned to user ({email})")
        Log().log(msg=f'Namespace ({namespace}) is not assigned to user ({email})')
        return False


def _check_email(email: str) -> bool:
    """
    Check whether email already exists

    :param email: str
        Email address

    :return: bool
        Whether email exists or not
    """
    return KUBEFLOW_USER_MANAGEMENT.is_registered(user_email=email)


def _check_namespace(namespace: str) -> bool:
    """
    Check whether kubernetes namespace already exists

    :param namespace: str
        Namespace

    :return: bool
        Whether namespace exists or not
    """
    _kubeflow_namespace: Dict[str, str] = KUBEFLOW_USER_MANAGEMENT.get_kubeflow_namespaces()
    _exists: bool = False
    for owner in _kubeflow_namespace.keys():
        if _kubeflow_namespace.get(owner) == namespace:
            _exists = True
            break
    return _exists


def _check_password(pwd: str) -> bool:
    """
    Check whether password meets the requirements

    :param pwd: str
        Password

    :return: bool
        Whether password meets the requirements
    """
    if len(pwd) < 30:
        st.error("Password must have at least 30 characters")
        Log().log(msg=f'Password must have at least 30 characters')
        return False
    if not re.search(r'\d', pwd):
        st.error("Password must contain numbers")
        Log().log(msg=f'Password must contain numbers')
        return False
    if not re.search(r'[!@#$%^&*()_+{}\[\]:;<>,.?~\\-]', pwd):
        st.error("Password must contain special characters")
        Log().log(msg=f'Password must contain special character')
        return False
    if not re.search(r'[A-Z]', pwd):
        st.error("Password must contain capital characters")
        Log().log(msg=f'Password must contain capital characters')
        return False
    if not re.search(r'[a-z]', pwd):
        st.error("Password must contain lower case characters")
        Log().log(msg=f'Password must contain lower case characters')
        return False
    return True


def _check_text_input(email: str, namespace: str, pwd: str) -> bool:
    """
    Check whether required text input field are filled out or not

    :param email: str
        Email address

    :param namespace: str
        Namespace

    :param pwd: str
        Password

    :return: bool
        Whether required text input field are filled out or not
    """
    if len(email) == 0:
        st.error(f"Email is empty")
        Log().log(msg=f'Email is empty')
        return False
    if len(namespace) == 0:
        st.error(f"Namespace is empty")
        Log().log(msg=f'Namespace is empty')
        return False
    if not re.search(r'^[a-z_-]+$', namespace):
        st.error("Namespace must contain only lower case characters, underscores and dashes")
        Log().log(msg=f'Namespace must contain only lower case characters, underscores and dashes')
        return False
    if len(pwd) == 0:
        st.error(f"Password is empty")
        Log().log(msg=f'Password is empty')
        return False
    return True


def _clear_text_input() -> None:
    """
    Clear text input fields
    """
    st.session_state['email'] = ""
    st.session_state['namespace'] = ""
    st.session_state['pwd'] = ""
    st.session_state['pwd_new'] = ""


def main() -> None:
    """
    Run Kubeflow user management (dex) streamlit application
    """
    st.set_page_config(page_title="Kubeflow User Management (Dex)")

    _col_1, _col_2 = st.columns([1, 2])

    with _col_1:
        st.image("kubeflow_icon.jpg", width=170, use_column_width=False)

    with _col_2:
        st.title("Kubeflow User Management")

    _email: str = st.text_input(label="Email", key='email')
    _namespace: str = st.text_input(label="Namespace", key='namespace')
    _password: str = st.text_input(label="Password", key='pwd', type="password")
    _new_password: str = st.text_input(label="New Password", key='pwd_new', type="password")

    _col_3, _col_4, _col_5 = st.columns([1, 2, 3])

    with _col_3:
        if st.button(label="Register"):
            if len(_new_password) > 0:
                if _check_text_input(email=_email, namespace=_namespace, pwd=_password):
                    if _check_password(pwd=_new_password):
                        KUBEFLOW_USER_MANAGEMENT.login_to_eks_cluster()
                        if _check_credentials(email=_email, namespace=_namespace, pwd=_password):
                            if KUBEFLOW_USER_MANAGEMENT.password_exists(pwd=_new_password):
                                st.error(f"Invalid password")
                                Log().log(msg=f'Invalid password')
                            else:
                                KUBEFLOW_USER_MANAGEMENT.replace_password(user_email=_email, new_pwd=_new_password)
                                st.success("Password of Kubeflow user changed")
                                Log().log(msg=f'Password of Kubeflow user ({_email}) changed')
                        else:
                            st.error("Invalid password")
                            Log().log(msg=f'Invalid password')
            else:
                if _check_text_input(email=_email, namespace=_namespace, pwd=_password):
                    if _check_password(pwd=_password):
                        KUBEFLOW_USER_MANAGEMENT.login_to_eks_cluster()
                        if _check_email(email=_email):
                            st.error(f"Email ({_email}) already registered")
                            Log().log(msg=f'Email ({_email}) already registered')
                        else:
                            if _check_namespace(namespace=_namespace):
                                st.error(f"Namespace ({_namespace}) already registered")
                                Log().log(msg=f'Namespace ({_namespace}) already registered')
                            else:
                                if KUBEFLOW_USER_MANAGEMENT.password_exists(pwd=_password):
                                    st.error(f"Invalid password")
                                    Log().log(msg=f'Invalid password')
                                else:
                                    KUBEFLOW_USER_MANAGEMENT.add_user(user_email=_email, user_name=_namespace, pwd=_password)
                                    st.success("Kubeflow user registered")
                                    Log().log(msg=f'Kubeflow user ({_email}) registered')

    with _col_4:
        if st.button(label="Delete"):
            if _check_text_input(email=_email, namespace=_namespace, pwd=_password):
                KUBEFLOW_USER_MANAGEMENT.login_to_eks_cluster()
                if _check_credentials(email=_email, namespace=_namespace, pwd=_password):
                    KUBEFLOW_USER_MANAGEMENT.delete_user(user_email=_email, user_name=_namespace)
                    st.success("Kubeflow user and namespace deleted")
                    Log().log(msg=f'Kubeflow user ({_email}) and namespace ({_namespace}) deleted')
                else:
                    st.error("Invalid password")
                    Log().log(msg=f'Invalid password')

    with _col_5:
        if st.button(label="Clear", on_click=_clear_text_input):
            Log().log(msg='Text input fields cleared')


if __name__ == "__main__":
    main()
