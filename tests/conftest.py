"""
Pytest configuration for AI Playground tests.
"""
import sys
from pathlib import Path

def pytest_configure(config):
    """Configure pytest with project root in path."""
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    print(f"Added to sys.path: {project_root}")
