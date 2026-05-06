"""Seed default clients and jobs into the database.

Idempotent — safe to call on every app boot. Reads the bundled JobSpec
JSON files in ``data/seed/`` and upserts them. New seed files dropped
into that directory are picked up automatically.
"""

from __future__ import annotations

import json
import logging
from importlib import resources

from .jobspec import JobSpec
from .storage import Storage

log = logging.getLogger(__name__)

DEFAULT_JOB_ID = "grupo-sazon/delivery-guy"


def _load_seeds() -> list[JobSpec]:
    """Read every ``*.json`` in ``hr_agent/data/seed/`` as a JobSpec."""
    out: list[JobSpec] = []
    seed_pkg = resources.files("hr_agent.data").joinpath("seed")
    for entry in seed_pkg.iterdir():
        if entry.name.endswith(".json"):
            with entry.open() as f:
                data = json.load(f)
            out.append(JobSpec.model_validate(data))
    return out


def ensure_default(storage: Storage) -> None:
    """Insert the seeded clients and jobs if they don't already exist.

    Upsert semantics: the spec_json is overwritten on each call. That's
    deliberate during development — fix the seed file, restart the app,
    the new spec is in. For production we'd switch to "insert if missing"
    and edit specs through an admin UI.
    """
    seeds = _load_seeds()
    for spec in seeds:
        storage.upsert_client(spec.client)
        storage.upsert_job(spec)
        log.info("seeded job: %s (client=%s)", spec.job_id, spec.client.id)
