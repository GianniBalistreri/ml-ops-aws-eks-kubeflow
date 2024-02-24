"""

Generate Secret Hash for authenticate to AWS Cognito

"""

import base64
import hashlib
import hmac
import os


def calculate_secret_hash(username: str, client_id: str, client_secret: str) -> str:
    """
    Calculate secret hash

    :param username: str
        Name of the user

    :param client_id: str
        Client id

    :param client_secret: str
        Client secret

    :return: str
        Base64 hashed secret
    """
    message = f'{username}{client_id}'
    dig = hmac.new(str(client_secret).encode('utf-8'),
                   msg=str(message).encode('utf-8'),
                   digestmod=hashlib.sha256
                   ).digest()
    return base64.b64encode(dig).decode()


if __name__ == '__main__':
    username = os.getenv('USERNAME')
    client_id = os.getenv('CLIENT_ID')
    client_secret = os.getenv('CLIENT_SECRET')
    secret_hash = calculate_secret_hash(username, client_id, client_secret)
    print(secret_hash)
