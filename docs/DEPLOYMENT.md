# Deployment, Monitoring & Scale

> Pragmatic plan for shipping this to Grupo Sazón. Premise: ~200 candidates/week today, design for 5×.

## Environments

| Env | Purpose | URL pattern |
|---|---|---|
| dev | Local laptop, SQLite, no live LLM (stub client in tests) | localhost |
| staging | Hosted, real LLM, throwaway DB, fed by synthetic candidates | screen-stg.gruposazon.com |
| prod | Live | screen.gruposazon.com |

Promotion: `main` → staging on every commit; tagged release → prod with manual approval.

## Topology

The shape that works at this scale:

```
            Candidate (WhatsApp / Web / SMS gateway)
                            │
                    ┌───────▼────────┐
                    │   API Gateway  │  TLS, rate limit, auth
                    └───────┬────────┘
                            │
              ┌─────────────▼─────────────┐
              │  Screening Agent service  │  (Python, FastAPI, 2-4 replicas)
              │  - stateless              │
              └─────────────┬─────────────┘
                            │
        ┌──────────────┬────┴────┬──────────────┐
        ▼              ▼         ▼              ▼
   Postgres       Anthropic   Redis          Object store
   (state +      (LLM)        (rate limit,   (transcript JSON
    transcripts)              dedupe keys)    cold archive)
```

**Why this is enough.** 200/week peaks at maybe ~5 concurrent conversations. Two replicas of a Python service handle that with room to spare. We avoid Kubernetes, queues, and microservices because the workload doesn't justify them.

**What's removable.** Redis is optional until we're managing hot-deduplication of incoming WhatsApp webhooks. Object store is optional — Postgres can hold transcripts at this volume.

## Caching analysis — do we need Redis?

Short answer: **not yet for caching, but yes eventually for other reasons.** I added it to `docker-compose.yml` under the `cache` profile (off by default, started with `docker compose --profile cache up`) so the topology is ready when the use case is real.

**Why caching is not the right framing today:**

| Candidate for caching | Verdict |
|---|---|
| Anthropic API responses | **Already cached.** We use the SDK's `cache_control: ephemeral` on the system prompt. Caching is server-side at Anthropic; Redis would not help and would risk staleness. |
| FAQ retrieval | 10 entries, in-memory dict scan, microseconds. Caching adds latency, not removes it. |
| Service-area validation | 45 cities + aliases, in-memory. Same as above. |
| Conversation reads | Postgres with the indexes we ship hits these in <2 ms at this volume. The bottleneck is the LLM (~1-3 s/turn), not the DB. |
| Analytics aggregations | Compute-on-read across ≤1k rows takes <50 ms. If a recruiter dashboard later polls every 5 s, *then* a 30-second cache wrapper around `analytics.compute()` is a 5-line change — but not required today. |

**When Redis earns its keep:**

1. **Webhook dedupe.** WhatsApp/SMS gateways retry deliveries. `SETNX` with TTL is the canonical idempotency primitive for "have I already processed message id X". Required the day we plug into a real channel.
2. **Per-candidate rate limiting.** A candidate hammering the chat (or a bug retrying on their side) needs a token bucket. Redis is the standard home.
3. **Distributed lock for re-engagement scheduler.** Once we add the "ping after 30 min / 24 h / 72 h silence" job, multi-replica coordination needs a lock — Redis `SET NX EX` is sufficient.
4. **Multi-instance session affinity** — only if we move conversation state out of Postgres into a hot store. Premature.

**Decision:** ship the compose file with Redis defined-but-not-running. The day we wire WhatsApp, we flip the profile and add ~50 lines of dedupe/rate-limit code. We don't run a service today that does nothing.

## Stack choices

| Concern | Choice | Why |
|---|---|---|
| Compute | Container on **Fly.io** or **Cloud Run** | Cheap, scale-to-zero; we're idle most of the time |
| DB | **Postgres 15** (managed: Neon / Cloud SQL) | We outgrow SQLite by month 1; same code, swap drivers |
| Secrets | Cloud secret manager (per env) | `ANTHROPIC_API_KEY`, ATS webhook secrets — never in repo |
| Observability | **OpenTelemetry → Grafana Cloud** | One stack for logs, metrics, traces |
| Error tracking | **Sentry** | Captures the long tail of weird candidate inputs |
| CI | GitHub Actions: lint → test → docker build → deploy | |

## Migration from take-home prototype

1. Replace `Storage` SQLite driver with asyncpg (interface is unchanged).
2. Move the `/api/conversations` REST surface behind a WhatsApp webhook adapter (same internal API).
3. Add Redis for dedupe of inbound message IDs (WhatsApp can replay).
4. Swap the static UI for a "recruiter console" — list, filter, summary view, manual override.

## Monitoring

### What we alert on

**P1 (page-on-call):**
- Anthropic 5xx rate > 5% over 5 min
- Screening service availability < 99% over 5 min
- DB primary down

**P2 (Slack channel):**
- Tool dispatch error rate > 1% over 15 min (= validator regressions or schema drift)
- Average tokens per conversation up 25% week-over-week (= prompt regression)
- Drop-off rate at any single stage doubles week-over-week

**P3 (weekly review):**
- Per-stage drop-off heatmap
- Decision mix (qualified / DQ-licence / DQ-zone / needs-review)
- Ratio of FAQ hits per conversation (proxy for prompt quality)

### Dashboards

- **Health:** RPS, p50/p95/p99 latency, error rate, Anthropic spend per day.
- **Funnel:** total → completed → qualified, broken down by country and stage.
- **Cost:** tokens/conversation, cost/conversation, cache hit rate.
- **Quality:** % conversations with `needs_review`, % with guardrail flags, manual override rate from recruiters.

## Cost

Rough numbers per screening with Claude Sonnet 4.6:

| Item | Tokens | $ |
|---|---|---|
| System prompt (cached after turn 1) | ~1.2k input | ~$0.001 first call, ~$0.0001 thereafter |
| Conversation history | growing 100-300/turn | linear |
| Response | ~50-150/turn | linear |
| Tool calls (round trip) | similar | |
| Summary | ~600 input + 300 output | ~$0.003 |

Per completed conversation: **~$0.04-0.08**. At 200/week: **~$10-15/week** in LLM costs. A single recruiter hour costs more than a month of agent runtime.

## Scaling levers (when we need them)

1. **Prompt caching** is on (`cache_control: ephemeral` on the system block) — first lever, already pulled.
2. **Haiku for the summary call** — same shape, ~5× cheaper, summary is structured JSON so accuracy delta is minimal.
3. **Hybrid model routing** — Haiku for the first 1-2 turns (license + city are simple), Sonnet for the rest. Probably premature.
4. **Batching analytics queries** — when the recruiter dashboard goes from polling to subscribing.
5. **Read replicas** for analytics once the conversation table crosses ~1M rows.

## Failure modes & runbook hooks

| Failure | Symptom | Playbook |
|---|---|---|
| Anthropic outage | 5xx rate spikes | Server returns 503 to UI; show "we're having a hiccup, your progress is saved"; conversations resume on next message. |
| Validator regression | `tool_dispatch_error` rate up | Roll back service; replay affected conversations via `/api/v1/screenings/{id}/replay`. |
| Prompt regression | Drop-off up at one stage | A/B between system prompt versions (we keep last 3 in `prompts.py`). |
| LLM cost runaway | Cost dashboard alert | Enforce per-conversation token budget (config: `MAX_TOKENS_PER_CONVERSATION`); end with `needs_review` when exceeded. |
| Candidate floods us with spam | One number, many conversations | Rate-limit per phone number at the gateway; existing Redis dedupe handles replay. |

## Security

- **Network:** API only on TLS 1.3+. Internal services on private VPC.
- **Secrets:** never in env files in repo; only in cloud secret manager. `.env.example` documents shape only.
- **PII:** transcripts stored in Postgres (encrypted at rest). Analytics exports redact via `guardrails.redact_pii`.
- **Audit:** every state mutation (tool dispatch) logged with conversation ID. Recruiter actions (manual override) tracked separately.
- **Right-to-deletion:** see ATS_INTEGRATION.md GDPR section.

## What I'd skip (for now)

- Kubernetes — overkill until we're at 10×.
- Vector DB for FAQ — 10 entries, keyword scoring is fine.
- Multi-region — Spain + Mexico can both serve from EU comfortably; latency is irrelevant when you're typing.
- Custom STT/TTS — browser Web Speech API for the prototype; ElevenLabs/Whisper if we go heavy on voice.
