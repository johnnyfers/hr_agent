import os
import tempfile
from pathlib import Path

import pytest

from hr_agent.storage import Storage


@pytest.fixture
def tmp_storage():
    """A throwaway SQLite DB per test."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "test.db"
        yield Storage(str(db))
