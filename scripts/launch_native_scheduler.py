"""Run a recorded scheduler revision against an unchanged frozen inference tree."""
import importlib.util
from pathlib import Path
import sys


def main():
    path = Path(sys.argv.pop(1)).resolve(strict=True)
    spec = importlib.util.spec_from_file_location('exp.baseline._scheduler_overlay', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.main()


if __name__ == '__main__':
    main()
