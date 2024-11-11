import yaml
import hashlib

def load_config(config_file='app\config.yaml'):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)


def get_hashed_key(secret_key: str) -> str:
    return hashlib.sha256(secret_key.encode()).hexdigest()


def verify_api_key(provided_key: str, hashed_key: str) -> bool:
    provided_hash = hashlib.sha256(provided_key.encode()).hexdigest()
    return provided_hash == hashed_key
