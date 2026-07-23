"""Import the CHO parmest model from the main checkout.

The CHO file (benchmarks/cho/parmest_nl_export.py) is NOT modified (task
constraint); we only add its directory to sys.path and import the model
builders. matplotlib is a plain dependency of that script and is installed
in the QA venv for this purpose.
"""
import sys

_CHO_DIR = "/Users/jkitchin/projects/pounce/benchmarks/cho"


def load():
    if _CHO_DIR not in sys.path:
        sys.path.insert(0, _CHO_DIR)
    import parmest_nl_export as cho  # noqa: E402
    return cho
