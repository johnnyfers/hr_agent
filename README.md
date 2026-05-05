# Grupo Sazón — Delivery Driver Screening Agent

A conversational AI screener for delivery driver applicants. Replaces phone-based pre-screening with an asynchronous chat that filters out unqualified candidates (~80% of recruiter time) and hands qualified ones to a human recruiter with a structured summary.

Built as a take-home assignment. Stack: Python · FastAPI · Anthropic Claude (Sonnet 4.6) · SQLite · vanilla JS chat UI with Web Speech voice.

---

## What's in here

| Path | What |
|---|---|
| [docs/PHASE1_DESIGN.md](docs/PHASE1_DESIGN.md) | Conversation flow, edge cases, tone guidelines (the design phase) |
| [docs/ATS_INTEGRATION.md](docs/ATS_INTEGRATION.md) | API spec for connecting to an ATS |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | How to deploy, monitor, scale |
| `src/hr_agent/agent.py` | LLM agent with tool-use screening flow |
| `src/hr_agent/schema.py` | Pydantic state models |
| `src/hr_agent/validators.py` | Deterministic validation (the trust boundary) |
| `src/hr_agent/prompts.py` | System prompt rendered with live state |
| `src/hr_agent/faq.py` | Lightweight FAQ retrieval (no vector DB needed at this scale) |
| `src/hr_agent/guardrails.py` | Pre-LLM input checks + PII redaction |
| `src/hr_agent/storage.py` | SQLite persistence |
| `src/hr_agent/analytics.py` | Funnel metrics |
| `src/hr_agent/server.py` | FastAPI surface |
| `static/index.html` | Browser chat UI (with Web Speech voice) |
| `tests/` | Unit + scenario tests with stubbed LLM |
| `scripts/run_simulation.py` | End-to-end eval — Claude-vs-Claude personas |

---

## Run it

```bash
# Python 3.11+
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=...

# Tests (no API key needed — LLM is stubbed):
pytest -q

# Server:
python -m hr_agent.server
# open http://localhost:8000

# End-to-end smoke test (uses real API, ~2-3¢):
python scripts/run_simulation.py --persona qualified
python scripts/run_simulation.py --persona all       # every persona
```

### Endpoints

| Method | Path | |
|---|---|---|
| `GET`  | `/`                                | Chat UI |
| `POST` | `/api/conversations`               | Start a screening |
| `POST` | `/api/conversations/{id}/turn`     | Send a user message |
| `GET`  | `/api/conversations/{id}`          | Fetch state + transcript |
| `POST` | `/api/conversations/{id}/summary`  | Generate recruiter summary |
| `GET`  | `/api/analytics`                   | Funnel metrics |

---

## Design decisions worth calling out

**LLM = Claude Sonnet 4.6.** Three reasons specific to this problem:
- **Multilingual ES/EN with code-switching** — Spain + Mexico, candidates mix languages mid-sentence; Sonnet handles it without translation steps.
- **Tool use is reliable** — every state change goes through a tool call, so the agent's structured output is the same primitive as its reasoning. No second-pass extraction.
- **Cost at scale** — ~$0.04-0.08 per completed screening with prompt caching on. At 200/week that's ~$10-15/week — orders of magnitude cheaper than a recruiter hour.

**State lives outside the LLM.** A Pydantic `ScreeningState` is rendered into every system prompt and mutated only by tool dispatch. The model proposes; deterministic validators dispose. This is the single most important architectural choice: it makes the system **testable** (we mock tool calls and assert state transitions, no LLM needed) and **safe** (a bad model output can't accidentally qualify a candidate without a licence).

**No vector DB for FAQ.** 10 entries. Keyword/tag scoring with diacritic normalisation works fine and is debuggable. We can swap in embeddings the day the FAQ grows past ~50 entries.

**Validation lives in `validators.py`, not the prompt.** The prompt encodes flow and tone. Field validity is data, and data belongs in code. This means validation is testable in milliseconds (see `tests/test_validators.py`) instead of probabilistic LLM evals.

**Hard filters first.** Licence and zone are the only disqualifying fields. We ask them before name. This is a 5-7 minute saving per unqualified candidate and is more honest than collecting their CV before saying no.

**Prompt caching.** The system prompt is the bulk of input tokens and is identical across turns within a conversation. It's marked as a cache breakpoint so each turn after the first reads it from cache.

**Guardrails are defence-in-depth.** Prompt injection patterns, length cap, off-topic detection — all happen *before* the LLM sees the message. The LLM also has guardrail rules in its system prompt. Either layer alone is insufficient.

**Disqualification etiquette.** When the agent says no, it says *why* and offers a soft next step (re-apply, alternative role). Adults handle a clear "no" better than vagueness. See `docs/PHASE1_DESIGN.md` §5.

---

## Bonus features delivered

- ✅ **Multi-language ES/EN** with mid-conversation code-switching (the agent records language as a field via tool call so persistence works correctly)
- ✅ **RAG over FAQ** — keyword retrieval with confidence threshold; the agent will say "the recruiter can confirm that" rather than invent answers
- ✅ **Guardrails** — prompt-injection detection, length cap, PII redaction for analytics exports
- ✅ **Analytics** — funnel metrics, drop-off-by-stage, average duration, qualification rate
- ✅ **Tests** — unit tests for validators, storage, guardrails, FAQ; scenario tests for the full agent loop with a stubbed Anthropic client; live eval script (`scripts/run_simulation.py`) with Claude-vs-Claude personas
- ✅ **ATS integration spec** — REST + webhook design in `docs/ATS_INTEGRATION.md`
- ✅ **Deployment design** — topology, monitoring, cost, runbook hooks in `docs/DEPLOYMENT.md`
- ✅ **Browser voice agent** — Web Speech API for STT/TTS in the included UI (works in Chrome/Edge)

Skipped: ElevenLabs voice (Web Speech is free and demonstrates the concept; ElevenLabs would be a 1-day add for production).

---

## How to evaluate this

The fastest read:
1. `docs/PHASE1_DESIGN.md` — what I'd build and why
2. `src/hr_agent/agent.py` — the heart of the system, ~300 lines
3. `tests/test_scenarios.py` — what behaviours I asserted
4. Run `python scripts/run_simulation.py --persona all` if you have an API key

Each persona exercises a different path: qualified, no-licence, out-of-zone, English-only, rude, asks-questions-first. The transcript prints to stdout along with tool calls and the final summary.
