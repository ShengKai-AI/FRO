import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
def asset(name):
    return Path(os.environ.get("FRO_" + name.upper(), ROOT / name))
def repo(name):
    return asset("third_party") / name
def model_path(name):
    return Path(os.environ.get("FRO_MODEL_" + name.upper(), asset("weights") / ("Qwen2.5-Omni-" + name)))
