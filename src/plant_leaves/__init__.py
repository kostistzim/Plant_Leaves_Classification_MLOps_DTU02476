import os
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
