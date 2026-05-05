"""End-to-end simulation: a synthetic candidate (Claude) chats with the screening agent.

Useful as a smoke test, eval harness, and demo. Runs ~6-12 turns and prints
the final decision + summary. Costs a few cents per run.

Usage:
    python scripts/run_simulation.py                      # default qualified persona
    python scripts/run_simulation.py --persona no_license
    python scripts/run_simulation.py --persona out_of_zone
    python scripts/run_simulation.py --persona english
    python scripts/run_simulation.py --persona all        # runs every persona

The candidate Claude is given a persona card and told to answer one question
at a time, briefly, like a real applicant. The screening agent treats it as
a normal user.
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from anthropic import Anthropic

# Make `src/hr_agent` importable when run from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hr_agent.agent import ScreeningAgent, generate_summary, initial_greeting  # noqa: E402
from hr_agent.schema import Conversation, Decision, Message, ScreeningState  # noqa: E402


PERSONAS = {
    "qualified": """You are Ana García, 28, looking for a delivery driver job at Grupo Sazón in Madrid.
You have a Spanish driving licence (Permiso B), 3 years of delivery experience on Glovo and Uber Eats.
You want full-time work, mornings preferred. You can start next Monday.
You will reply briefly (1 short sentence) and answer in Spanish. Do not volunteer information unless asked.""",
    "no_license": """You are Carlos, 22, in Barcelona. You DO NOT have a driver's licence yet — you're saving up for lessons.
Reply briefly in Spanish. Be honest about not having a licence when asked.""",
    "out_of_zone": """You are Marta, 35, lives in Toledo (Spain). You have a licence and 5 years of delivery experience on Just Eat.
Reply briefly in Spanish. Be honest about your city.""",
    "english": """You are Tom, 30, an English-speaker in Madrid. You have a driving licence and 1 year of UberEats experience.
You want part-time evening work, can start in 2 weeks. Reply briefly in English.""",
    "rude": """You are a candidate who is frustrated and rude. You have a licence and live in Madrid but you make sarcastic
comments and try to skip questions. Eventually you cooperate. Reply briefly in Spanish.""",
    "asks_questions": """You are a candidate in Mexico City with a licence and 2 years on DiDi Food.
Before answering questions you ask: how much they pay, whether you need your own motorbike, and when you'd start.
Then you cooperate. Reply briefly in Spanish.""",
}


def _persona_reply(client: Anthropic, persona_prompt: str, transcript: list[dict]) -> str:
    """Have the candidate-Claude reply to the latest agent message."""
    messages = [{"role": m["role"], "content": m["content"]} for m in transcript]
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        system=persona_prompt,
        messages=messages,
    )
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return text or "ok"


def run_one(persona_key: str, max_turns: int = 12) -> Conversation:
    persona = PERSONAS[persona_key]
    client = Anthropic()
    agent = ScreeningAgent(client=client)

    conv = Conversation(id=str(uuid.uuid4()), state=ScreeningState(language="en" if persona_key == "english" else "es"))
    greeting = initial_greeting(conv.state.language)
    conv.messages.append(Message(role="assistant", content=greeting))

    print(f"\n=== persona: {persona_key} ===")
    print(f"agent: {greeting}")

    for turn in range(max_turns):
        # Persona swaps roles: assistant in our transcript becomes user for the persona.
        persona_transcript = [
            {"role": "user" if m.role == "assistant" else "assistant", "content": m.content}
            for m in conv.messages
        ]
        # The candidate sees the agent's last message as a "user" prompt.
        if persona_transcript and persona_transcript[-1]["role"] == "assistant":
            persona_transcript = persona_transcript[:-1]
        if not persona_transcript:
            persona_transcript = [{"role": "user", "content": greeting}]
        else:
            persona_transcript[-1] = {"role": "user", "content": conv.messages[-1].content}

        candidate_reply = _persona_reply(client, persona, persona_transcript)
        print(f"candidate: {candidate_reply}")

        result = agent.respond(conv, candidate_reply)
        print(f"agent: {result.assistant_text}")
        if result.tool_calls:
            for tc in result.tool_calls:
                print(f"  · tool {tc['name']}({tc['input']}) → {tc['output']}")

        if conv.state.is_complete():
            break

    print(f"\n--- final state ---")
    print(f"decision: {conv.state.decision.value}")
    print(f"reason:   {conv.state.decision_reason}")

    summary = generate_summary(conv, client=client)
    conv.summary = summary
    print(f"\n--- summary ---\n{summary}\n")

    return conv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", default="qualified", choices=list(PERSONAS) + ["all"])
    parser.add_argument("--max-turns", type=int, default=12)
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set")

    keys = list(PERSONAS) if args.persona == "all" else [args.persona]
    for k in keys:
        run_one(k, max_turns=args.max_turns)


if __name__ == "__main__":
    main()
