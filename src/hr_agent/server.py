"""FastAPI surface — REST endpoints for chat + a minimal browser UI.

Endpoints:
- GET  /api/clients                              → list registered clients
- GET  /api/jobs                                 → list registered jobs
- GET  /api/jobs/{job_id:path}                   → fetch one JobSpec
- POST /api/conversations                        → start a screening (body: job_id, language)
- POST /api/conversations/{id}/turn              → send a user message
- GET  /api/conversations/{id}                   → fetch state + transcript
- POST /api/conversations/{id}/summary           → generate recruiter summary
- GET  /api/analytics                            → funnel metrics (?job_id= optional)
- GET  /                                         → static chat UI
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Optional

from anthropic import Anthropic
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from . import analytics
from .agent import ScreeningAgent, generate_summary, initial_greeting
from .schema import Conversation, Message, ScreeningState
from .seed import DEFAULT_JOB_ID, ensure_default
from .storage import Storage, make_storage

log = logging.getLogger(__name__)
logging.basicConfig(
    level=os.environ.get("HR_AGENT_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


def _find_static_dir() -> Optional[Path]:
    """Locate the static/ directory across install layouts.

    - editable/source: server.py at src/hr_agent/server.py → ../../static
    - Docker (non-editable install): server.py is in site-packages but the
      Dockerfile copies static to /app/static and runs uvicorn from /app
    - explicit override: HR_AGENT_STATIC_DIR
    """
    override = os.environ.get("HR_AGENT_STATIC_DIR")
    if override:
        p = Path(override)
        return p if p.exists() else None

    candidates = [
        Path(__file__).resolve().parent.parent.parent / "static",  # src layout
        Path.cwd() / "static",                                      # Docker WORKDIR
        Path("/app/static"),                                        # Docker explicit
    ]
    for c in candidates:
        if c.exists() and (c / "index.html").exists():
            return c
    return None


def _build_app(storage: Optional[Storage] = None, agent: Optional[ScreeningAgent] = None) -> FastAPI:
    """Factory so tests can inject in-memory dependencies."""
    app = FastAPI(title="Multi-tenant Screening Agent", version="0.2.0")
    app.state.storage = storage or make_storage()
    app.state.client = Anthropic() if os.environ.get("ANTHROPIC_API_KEY") else None
    app.state.agent = agent or (ScreeningAgent(client=app.state.client) if app.state.client else None)

    # Seed default clients + jobs (idempotent).
    try:
        ensure_default(app.state.storage)
    except Exception as e:
        log.exception("failed to seed default jobs: %s", e)

    static_dir = _find_static_dir()
    if static_dir is not None:
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/")
        def root():
            return FileResponse(str(static_dir / "index.html"))
    else:
        log.warning("static UI directory not found — chat UI disabled")

    _register_routes(app)
    return app


# --- Request / response models ---------------------------------------------


class StartRequest(BaseModel):
    candidate_id: Optional[str] = None
    language: str = "es"
    job_id: str = DEFAULT_JOB_ID


class StartResponse(BaseModel):
    conversation_id: str
    job_id: str
    greeting: str


class TurnRequest(BaseModel):
    message: str


class TurnResponse(BaseModel):
    reply: str
    state: ScreeningState
    done: bool
    guardrail_flag: Optional[str] = None


# --- Routes -----------------------------------------------------------------


def _register_routes(app: FastAPI) -> None:
    storage: Storage = app.state.storage

    @app.get("/api/clients")
    def list_clients():
        return [c.model_dump() for c in storage.list_clients()]

    @app.get("/api/jobs")
    def list_jobs(client_id: Optional[str] = None):
        jobs = storage.list_jobs(client_id=client_id)
        return [
            {
                "job_id": j.job_id,
                "client": j.client.model_dump(),
                "title": j.title,
                "description": j.description,
                "languages": j.languages,
                "field_count": len(j.fields),
            }
            for j in jobs
        ]

    @app.get("/api/jobs/{job_id:path}")
    def get_job(job_id: str):
        job = storage.get_job(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        return job.model_dump()

    @app.post("/api/conversations", response_model=StartResponse)
    def start(req: StartRequest):
        job = storage.get_job(req.job_id)
        if not job:
            raise HTTPException(404, f"job not found: {req.job_id}")
        lang = req.language if req.language in job.languages else job.default_language
        conv_id = str(uuid.uuid4())
        state = ScreeningState(
            job_id=job.job_id,
            client_id=job.client.id,
            language=lang,
        )
        greeting = initial_greeting(job, lang)
        conv = Conversation(
            id=conv_id,
            candidate_id=req.candidate_id,
            state=state,
            messages=[Message(role="assistant", content=greeting)],
        )
        storage.upsert_conversation(conv)
        for m in conv.messages:
            storage.append_message(conv_id, m)
        return StartResponse(conversation_id=conv_id, job_id=job.job_id, greeting=greeting)

    @app.post("/api/conversations/{conv_id}/turn", response_model=TurnResponse)
    def turn(conv_id: str, req: TurnRequest):
        conv = storage.get_conversation(conv_id)
        if not conv:
            raise HTTPException(404, "conversation not found")
        if conv.state.is_complete():
            raise HTTPException(409, "conversation already complete")
        if app.state.agent is None:
            raise HTTPException(503, "agent not configured (missing ANTHROPIC_API_KEY)")

        job = storage.get_job(conv.state.job_id)
        if not job:
            raise HTTPException(500, f"job spec missing for {conv.state.job_id}")

        pre_count = len(conv.messages)
        result = app.state.agent.respond(conv, req.message, job)

        for m in conv.messages[pre_count:]:
            storage.append_message(conv_id, m)
        storage.upsert_conversation(conv)

        return TurnResponse(
            reply=result.assistant_text,
            state=conv.state,
            done=conv.state.is_complete(),
            guardrail_flag=result.guardrail_flag,
        )

    @app.get("/api/conversations/{conv_id}")
    def get_conv(conv_id: str):
        conv = storage.get_conversation(conv_id)
        if not conv:
            raise HTTPException(404, "conversation not found")
        return conv.model_dump(mode="json")

    @app.post("/api/conversations/{conv_id}/summary")
    def summarize(conv_id: str):
        conv = storage.get_conversation(conv_id)
        if not conv:
            raise HTTPException(404, "conversation not found")
        job = storage.get_job(conv.state.job_id)
        if not job:
            raise HTTPException(500, f"job spec missing for {conv.state.job_id}")
        summary = generate_summary(conv, job, client=app.state.client)
        conv.summary = summary
        storage.upsert_conversation(conv)
        return {"summary": summary}

    @app.get("/api/analytics")
    def get_analytics(job_id: Optional[str] = None):
        m = analytics.compute(storage, job_id=job_id)
        return {
            "metrics": m.__dict__,
            "stage_drop_off_table": analytics.stage_drop_off_table(m),
            "scoped_to_job": job_id,
        }

    @app.get("/api/health")
    def health():
        return {"ok": True, "agent_configured": app.state.agent is not None}


app = _build_app()


def main():
    import uvicorn

    uvicorn.run("hr_agent.server:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
