import yaml

def load_config(path: str) -> dict:
    """
    Load YAML config file into a dictionary.
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg
