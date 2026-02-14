"""Pytest shared setup for local module imports.

Ensures repository root modules (e.g. ``strategies.py``) are importable even
when pytest is invoked from outside the project root.
"""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)
