"""FastAPI surface — REST endpoints for chat + a minimal browser UI.

Endpoints:
- POST /api/conversations            → start a new screening
- POST /api/conversations/{id}/turn  → send a user message, get a reply
- GET  /api/conversations/{id}       → fetch full state + transcript
- POST /api/conversations/{id}/summary → generate recruiter summary
- GET  /api/analytics                → funnel metrics
- GET  /                              → static chat UI
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
from .storage import Storage

log = logging.getLogger(__name__)
logging.basicConfig(
    level=os.environ.get("HR_AGENT_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


# --- App lifecycle ----------------------------------------------------------


def _build_app(storage: Optional[Storage] = None, agent: Optional[ScreeningAgent] = None) -> FastAPI:
    """Factory so tests can inject in-memory dependencies."""
    app = FastAPI(title="Grupo Sazón Screening Agent", version="0.1.0")
    app.state.storage = storage or Storage()
    app.state.client = Anthropic() if os.environ.get("ANTHROPIC_API_KEY") else None
    app.state.agent = agent or (ScreeningAgent(client=app.state.client) if app.state.client else None)

    static_dir = Path(__file__).parent.parent.parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/")
        def root():
            return FileResponse(str(static_dir / "index.html"))

    _register_routes(app)
    return app


# --- Request / response models ---------------------------------------------


class StartRequest(BaseModel):
    candidate_id: Optional[str] = None
    language: str = "es"


class StartResponse(BaseModel):
    conversation_id: str
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

    @app.post("/api/conversations", response_model=StartResponse)
    def start(req: StartRequest):
        lang = req.language if req.language in ("es", "en") else "es"
        conv_id = str(uuid.uuid4())
        state = ScreeningState(language=lang)
        greeting = initial_greeting(lang)
        conv = Conversation(
            id=conv_id,
            candidate_id=req.candidate_id,
            state=state,
            messages=[Message(role="assistant", content=greeting)],
        )
        storage.upsert_conversation(conv)
        for m in conv.messages:
            storage.append_message(conv_id, m)
        return StartResponse(conversation_id=conv_id, greeting=greeting)

    @app.post("/api/conversations/{conv_id}/turn", response_model=TurnResponse)
    def turn(conv_id: str, req: TurnRequest):
        conv = storage.get_conversation(conv_id)
        if not conv:
            raise HTTPException(404, "conversation not found")
        if conv.state.is_complete():
            raise HTTPException(409, "conversation already complete")
        if app.state.agent is None:
            raise HTTPException(503, "agent not configured (missing ANTHROPIC_API_KEY)")

        # Snapshot pre-turn message count so we know what's new.
        pre_count = len(conv.messages)
        result = app.state.agent.respond(conv, req.message)

        # Persist new messages (user + assistant) and the updated state.
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
        summary = generate_summary(conv, client=app.state.client)
        conv.summary = summary
        storage.upsert_conversation(conv)
        export_path = storage.export_json(conv_id)
        return {"summary": summary, "export_path": str(export_path)}

    @app.get("/api/analytics")
    def get_analytics():
        m = analytics.compute(storage)
        return {
            "metrics": m.__dict__,
            "stage_drop_off_table": analytics.stage_drop_off_table(m),
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
