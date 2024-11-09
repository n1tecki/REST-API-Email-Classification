import yaml
import hashlib

def load_config(config_file='app\config.yaml'):
    """Load configuration from a YAML file."""
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def get_hashed_key(secret_key: str) -> str:
    """Hash the provided secret key using SHA-256."""
    return hashlib.sha256(secret_key.encode()).hexdigest()

def verify_api_key(provided_key: str, hashed_key: str) -> bool:
    """Verify the provided API key against the hashed key."""
    provided_hash = hashlib.sha256(provided_key.encode()).hexdigest()
    return provided_hash == hashed_key
