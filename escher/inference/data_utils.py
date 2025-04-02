import json
import pickle
import re

import yaml

find_json_block = re.compile(r"```json(.*?)```", re.DOTALL)


def load_obj(pth):
    ext = pth.split(".")[-1]
    if ext == "pkl":
        with open(pth, "rb") as f:
            return pickle.load(f)
    elif ext == "json":
        with open(pth, "r") as f:
            return json.load(f)
    elif ext == "yaml":
        with open(pth, "r") as f:
            return yaml.safe_load(f)
    elif ext == "txt":
        with open(pth, "r") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported extension: {ext}, for file: {pth}")


def save_obj(obj, pth):
    ext = pth.split(".")[-1]
    if ext == "pkl":
        with open(pth, "wb") as f:
            pickle.dump(obj, f)
    elif ext == "json":
        with open(pth, "w") as f:
            json.dump(obj, f)
    elif ext == "yaml":
        with open(pth, "w") as f:
            yaml.dump(obj, f)
    elif ext == "txt":
        with open(pth, "w") as f:
            f.write(obj)
    else:
        raise ValueError(f"Unsupported extension: {ext}, for file: {pth}")
