import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

def read_text(file_path: str) -> str:
    with open(file_path, 'r') as file:
        return file.read()


def load_env(env_file_path: str = ".env") -> None: 
    env_file = read_text(env_file_path)
    for line in env_file.split("\n"):
        if line.strip() == "":
            continue
        key = line.split("=")[0].strip()
        value = "=".join(line.split("=")[1:]).strip()
        os.environ[key] = value
