# ATS Integration — API Spec

> Design only. The screening agent emits structured candidate records that any modern ATS can consume. We define the contract here so the integration is a 1-day plumbing job, not a re-architecture.

## Integration model

There are two practical patterns. We recommend **(B)**.

**(A) Push (webhook)** — agent posts to ATS when a screening terminates.
**(B) Pull (REST + webhook signal)** — ATS subscribes to a "screening completed" webhook (notification only) and pulls the canonical record from our API. Reasoning: ATS retries idempotently, and we keep the record-of-truth.

```
Candidate ↔ Screening Agent ─┬─▶ POST /webhooks/screening.completed (signal)
                             │
ATS ─── GET /api/v1/screenings/{id} ─┘  (pull canonical record)
```

## Auth

- **API key** in `Authorization: Bearer <key>`, scoped per ATS tenant.
- Webhook signatures: `X-Sazon-Signature: sha256=<hex(hmac(secret, body))>`. ATS verifies before acting.
- Rotation: keys rotatable from the admin panel; webhooks tolerate two simultaneously valid secrets for 7 days.

## Resources

### `Screening`

```json
{
  "id": "uuid",
  "candidate_id": "string|null",        // ATS-side id if already known
  "decision": "qualified | disqualified_no_license | disqualified_out_of_zone | needs_review | dropped_off",
  "decision_reason": "string|null",
  "language": "es | en",
  "candidate": {
    "full_name": "string",
    "has_license": true,
    "city": "Madrid",
    "country": "ES",
    "city_in_service_area": true,
    "availability": "full_time | part_time | weekends_only | flexible",
    "preferred_schedule": "morning | afternoon | evening | night | flexible",
    "experience": { "years": 3, "platforms": ["Glovo", "Uber Eats"] },
    "start_date": "2026-05-19"          // best-effort ISO if normalisable
  },
  "summary": "Recruiter-facing markdown summary",
  "transcript": [                        // optional, expand=transcript
    { "role": "assistant", "content": "...", "timestamp": "..." }
  ],
  "stage_reached": "8_confirmed",
  "needs_review_fields": [],
  "created_at": "2026-05-05T10:00:00Z",
  "updated_at": "2026-05-05T10:07:23Z"
}
```

### Webhook payload (notification only)

```json
{
  "event": "screening.completed",
  "screening_id": "uuid",
  "decision": "qualified",
  "occurred_at": "2026-05-05T10:07:23Z"
}
```

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST`   | `/api/v1/screenings` | Create a screening session (returns ID + opaque chat URL or starts a pre-filled session if you already collected the candidate) |
| `GET`    | `/api/v1/screenings/{id}` | Canonical record (above). `?expand=transcript` includes message log |
| `GET`    | `/api/v1/screenings?decision=qualified&since=2026-05-01` | Paginated list, filters by decision, country, date |
| `POST`   | `/api/v1/screenings/{id}/replay` | Re-run summary generation (idempotent, useful after model upgrades) |
| `POST`   | `/api/v1/webhooks` | Register a webhook URL + filter (decision, country) |
| `DELETE` | `/api/v1/webhooks/{id}` | |

### Pagination

Cursor-based (`?cursor=…&limit=50`). Stable order by `updated_at DESC, id DESC`. Max limit 200.

### Errors

Standard problem-detail JSON. Codes we'll actually emit:

| HTTP | Code | Meaning |
|---|---|---|
| 400 | `validation_error` | Malformed request body |
| 401 | `unauthorized` | Missing/bad API key |
| 404 | `not_found` | Screening or webhook ID unknown |
| 409 | `conflict` | Trying to mutate a completed screening |
| 429 | `rate_limited` | Exceeded tenant quota; `Retry-After` header set |
| 503 | `agent_unavailable` | LLM upstream is down — retry with backoff |

## Idempotency

- All write endpoints accept `Idempotency-Key` header (recommended for `POST /screenings`).
- Webhook deliveries retry 5× with exponential backoff (1m → 1h). After exhaustion, deliveries are queued in a DLQ and surfaced in the admin UI.

## Data lifecycle / GDPR

- Default retention: **180 days** post-completion. Configurable per tenant.
- `DELETE /api/v1/screenings/{id}` performs a hard delete (transcript + state + summary). Audit log retained 30 days.
- PII redaction is opt-in for analytics exports — see `guardrails.redact_pii`.
- Right-to-access: `GET /api/v1/candidates/{id}/data` returns a candidate's complete record across all screenings.

## What we deliberately don't do

- We don't push directly into ATS workflows (e.g. moving cards on Greenhouse pipelines). That's the ATS's job — it consumes the webhook and decides what to do. Keeps integration simple and avoids each ATS's quirks.
- We don't sync candidate updates back from the ATS. If a recruiter edits the candidate's info in the ATS, that's their record-of-truth from then on; our screening record is immutable post-completion.
- No GraphQL. REST is sufficient at this scale and easier for ops to debug.
