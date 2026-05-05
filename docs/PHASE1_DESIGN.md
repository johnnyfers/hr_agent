# Phase 1 — Conversation Design

**Client:** Grupo Sazón · **Role:** Delivery Driver · **Channel:** Messaging (chat-first, voice-capable)

## 1. Goals & Constraints

The agent replaces an outbound phone screen that fails 60% of the time. Two design implications follow:

1. **Asynchronous-friendly.** Candidates may answer over hours or days. The agent must tolerate gaps without re-introducing itself or losing state.
2. **Drop-off is the default.** ~60% of candidates already vanish on the phone. We optimise for **completion of the cheap fields first**, so even partial conversations have triage value.

We never disqualify silently — every "no" is explained and offers next steps (re-apply later, alternative role, etc.).

## 2. Conversation Stages

The flow is a soft state machine. The LLM owns wording; a Pydantic state object owns *what's been collected* and gates progression.

```
[0] Greet & consent
       ↓
[1] Hard filter:  Driver's licence?  ──no──▶  [D1] Disqualify (no licence)
       ↓ yes
[2] Hard filter:  Service area?      ──out──▶ [D2] Disqualify (out of zone)
       ↓ in
[3] Identity:     Full name
       ↓
[4] Availability: Full / part / weekends
       ↓
[5] Schedule:     Morning / afternoon / evening / flexible
       ↓
[6] Experience:   Years + platforms
       ↓
[7] Start date
       ↓
[8] Confirm summary + answer FAQ
       ↓
[Q] Qualified — handoff
```

**Why hard filters first.** Licence and zone are the only fields that can outright reject. Asking them up front saves 5-7 minutes of agent time per unqualified candidate, and is honest with the candidate (no sunk-cost feeling).

**Why name comes after the filters.** Driver's licence is impersonal and easy to answer; opening with "what's your full name?" feels like a form. We lead with the question that tells the candidate this is a real screening, not spam.

## 3. Data Fields & Validation

| # | Field | Type | Validation | On invalid |
|---|---|---|---|---|
| 1 | `has_license` | bool | yes/no/equivalents in ES & EN | Re-ask once, then accept free-text and let LLM resolve |
| 2 | `city` / `zone` | enum | must match `service_areas.json` (45 cities, fuzzy match) | Offer nearest covered city; if none within ~30km, disqualify |
| 3 | `full_name` | str | ≥2 tokens, letters/spaces/hyphens/apostrophes, ≤80 chars | Re-ask, give example |
| 4 | `availability` | enum | full_time / part_time / weekends_only / flexible | Re-ask with explicit options |
| 5 | `preferred_schedule` | enum | morning / afternoon / evening / night / flexible | Same |
| 6 | `experience_years` | int 0-40 | numeric, "none"/"ninguna" → 0 | Re-ask with examples |
| 6b | `experience_platforms` | list[str] | optional; common: Glovo, Uber Eats, Rappi, Just Eat, DoorDash | Free text accepted |
| 7 | `start_date` | str | "immediately", ISO date, or relative ("in 2 weeks") — agent normalises | Re-ask with examples |

**Validation policy.** All validation happens via tool calls (`record_field`, `validate_city`). The LLM proposes, the validator disposes. This keeps the LLM honest and gives us deterministic disqualification logic that is testable independently of model output.

## 4. Edge Cases

### 4.1 Candidate stops responding
- **Re-engagement after 30 min, 24 h, 72 h** (configurable, designed not built).
- Each follow-up references the last collected field so it's clearly a continuation, not spam.
- After the third silent ping, mark `dropped_off` with the last completed stage. This is *valuable analytics data* even though the candidate never finished.

### 4.2 Invalid / ambiguous answers
- **One clarifying re-ask per field**, then accept free-text and let the LLM extract.
- If extraction confidence is low, the agent flags the field as `needs_human_review` rather than guessing — recruiter sees it in the summary.
- Never trap the candidate: a third attempt always moves on.

### 4.3 Language switching (ES ↔ EN)
- First user message determines initial language (Claude detects in-context).
- The agent *follows* the candidate: if they switch, the agent switches on the next turn. Code-switching ("Tengo un Toyota, what year do you need?") is handled — the agent responds in the dominant language of the latest message.
- Saved data is always normalised to canonical English keys (`has_license`, `availability`) regardless of conversation language. Free-text fields (city, platforms) preserve original casing.

### 4.4 Off-topic / inappropriate input
- Hard refusal pattern for: legal/medical advice, attempts to renegotiate pay/role, abuse, prompt injection ("ignore previous instructions").
- Soft redirect for: chitchat, repeated questions about the company → answered briefly via FAQ, then steered back.
- See `guardrails.py` for full list.

### 4.5 Candidate asks questions
- FAQ retrieval is a tool call. Agent answers from FAQ only — if no match, says "let me note that for the recruiter" and continues. Never invents company facts.

## 5. Outcomes

| Outcome | Trigger | Candidate sees | System does |
|---|---|---|---|
| **Qualified** | All required fields collected, no disqualifiers | "Thanks — a recruiter from Grupo Sazón will reach out within 48h. Confirmation: [summary]." | Save state with `decision=qualified`, push to ATS handoff queue |
| **Disqualified — no licence** | `has_license=false` | "Driver's licence is required for this role. We'll keep your details on file in case other roles open up." | Save with `decision=disqualified_no_license`. Do not pretend they may pass later |
| **Disqualified — out of zone** | `city` not in service areas | "We don't currently deliver in {city}. We'll let you know if we expand." | Save with `decision=disqualified_out_of_zone` and the candidate's claimed city for expansion analytics |
| **Dropped off** | No reply for 72h after last re-engagement | (no message) | Save with `decision=dropped_off`, `last_stage=<n>` |
| **Needs review** | Validator confidence low, or recruiter override flagged | "I'll have a recruiter follow up to clarify a couple of details." | Save with `decision=needs_review`, list ambiguous fields |

## 6. Tone & Length Guidelines

This is messaging, not email. The bar:

- **Length:** ≤2 sentences per turn, except summaries. One question at a time.
- **Register:** Friendly-professional. Spanish defaults to *tú* (informal); the agent can switch to *usted* if the candidate clearly does first.
- **Empathy:** Acknowledge the candidate before pivoting. "Got it" / "perfecto" beats "Thank you for that information."
- **No bureaucratic language.** Not "please indicate your residential location" — "what city are you in?".
- **No emoji unless the candidate uses them first** (then sparingly — one per ~5 turns).
- **No false urgency** ("act now!"). Real urgency only — "we're filling spots this week, so a quick reply helps."
- **Never claim to be human.** If asked, the agent says it's an automated screener and a recruiter follows up later.

## 7. Open Questions for Grupo Sazón

(Designed-into-the-system but flagged for client confirmation.)

1. **Service area resolution.** We assume city-level granularity. If they need zone-level (e.g. specific Mexico City colonias), validator data needs to be more granular.
2. **License equivalence across countries.** Spanish *Permiso B* vs. Mexican *Licencia tipo A* — are both accepted? Currently treated as equivalent; flag for legal.
3. **Minimum age / right to work.** Not in the spec; we don't ask. If required, add as a hard filter after licence.
4. **Re-application policy.** A candidate disqualified on zone today may move tomorrow. Default: allow re-screening after 90 days; configurable.
