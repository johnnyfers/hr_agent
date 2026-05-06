import tempfile
from pathlib import Path

import pytest

from hr_agent.seed import _load_seeds, ensure_default
from hr_agent.storage import SqliteStorage


@pytest.fixture
def tmp_storage():
    """A throwaway SQLite DB per test, pre-seeded with the default jobs."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "test.db"
        storage = SqliteStorage(str(db))
        ensure_default(storage)
        yield storage


@pytest.fixture
def grupo_sazon_spec():
    """Return the seeded Grupo Sazón delivery-guy JobSpec without touching DB."""
    seeds = _load_seeds()
    return next(s for s in seeds if s.job_id == "grupo-sazon/delivery-guy")
