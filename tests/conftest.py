# tests/conftest.py  (update or create)
import sys
from pathlib import Path
import pytest

# -- Add project root to import path ---------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# -- Optional: register a custom CLI flag ----------------------
def pytest_addoption(parser):
    parser.addoption(
        "--gui", action="store_true", default=False,
        help="Run GUI-integration tests that move the real mouse."
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "gui: mark test as requiring real GUI clicks")
