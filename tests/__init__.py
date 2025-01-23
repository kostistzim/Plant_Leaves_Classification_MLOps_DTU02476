import os
from pathlib import Path

_TEST_ROOT = os.path.dirname(__file__)  # root of test folder
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)  # root of project
_PATH_DATA = Path("data/processed/")  # root of data
_PATH_TEST_DATA = Path("tests/test_data/")  # root of test data