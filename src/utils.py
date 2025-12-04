import os
import random
import numpy as np
from typing import List
import json


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


def write_jsonl(data: List[dict], file_path: str) -> None:
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')


def write_json(data: dict, file_path: str) -> None:
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)