"""Run sample exchanges against the live persona prompt and check how Jasper sounds.

Calls the model directly with the conscious_agent instructions and only the
ui_output tool, building each event exactly like agent_events does (recent
conversation + recently used puns), so no robot or server is needed.

    set -a; . ./.env; set +a
    ROBOT_NAME=Jasper venv/bin/python scripts/persona_check.py

On the robot (Azure, same model as the running agent):

    source setmyenv.sh && python scripts/persona_check.py

Reports pun rate, self-descriptions, replies over three sentences, and replies
the TTS cleaner had to change. Exits 1 if any check fails.
"""

import json
import os
import re
import sys
from pathlib import Path

from openai import AzureOpenAI
from openai import OpenAI
from pyhocon import ConfigFactory

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from coded_tools.unigo2.agent_events import _PUN_PATTERNS  # noqa: E402
from coded_tools.unigo2.agent_events import TurnContext  # noqa: E402
from coded_tools.unigo2.ui_output import speakable  # noqa: E402

TURNS = [
    "Hi there!",
    "How are you doing today?",
    "What did you do this morning?",
    "What's the capital of Australia?",
    "What are you, exactly?",
    "Are you a good boy?",
    "I had a pretty rough day. My project got cancelled.",
    "Thanks, that actually helps.",
    "Can you explain in detail how a heat pump works? Take your time.",
    "Great, thanks! See you later.",
]
SELF_DESCRIPTION = re.compile(r"(?i)\b(cute|adorable|good boy|robot dog)\b")
SENTENCE_END = re.compile(r"(?<!\.)[.!?](?=\s|$)")  # "..." is a pause, not an end
SERIOUS = {6}  # the sad turn: no pun allowed


def load_agent():
    config = ConfigFactory.parse_file(str(REPO_ROOT / "registries" / "conscious_agent.hocon"))
    tools = config["tools"]
    agent = next(tool for tool in tools if tool["name"] == "conscious_agent")
    ui_output = next(tool for tool in tools if tool["name"] == "ui_output")
    model = os.environ.get("AGENT_LLM_MODEL_NAME", config["llm_config"]["model_name"])
    tool = {"type": "function", "function": {
        "name": "ui_output",
        "description": ui_output["function"]["description"],
        "parameters": json.loads(json.dumps(ui_output["function"]["parameters"])),
    }}
    return agent["instructions"], tool, model


def ask(client, model, instructions, tool, event_text):
    # Same reasoning effort as the agent (setmyenv sets "none" for GPT-6).
    effort = os.environ.get("AGENT_LLM_REASONING_EFFORT")
    response = client.chat.completions.create(
        model=model,
        **({"reasoning_effort": effort} if effort else {}),
        messages=[{"role": "system", "content": instructions}, {"role": "user", "content": event_text}],
        tools=[tool],
    )
    message = response.choices[0].message
    for call in message.tool_calls or []:
        return json.loads(call.function.arguments).get("say", "")
    return message.content or ""


def main():
    instructions, tool, model = load_agent()
    # Azure (as on the robot) reads its key, endpoint and API version from the environment.
    azure = bool(os.environ.get("AZURE_OPENAI_ENDPOINT"))
    client = AzureOpenAI() if azure else OpenAI()
    if azure:
        model = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
    context = TurnContext()
    rows = []
    for number, text in enumerate(TURNS):
        event_text = f"user: {text}"
        block = context.conversation_block()
        if block:
            event_text += "\n" + block
        context.remember("user", text)
        raw = ask(client, model, instructions, tool, event_text)
        spoken = speakable(raw)
        context.remember("you", spoken)
        puns = [pun for pun, pattern in _PUN_PATTERNS if pattern.search(spoken)]
        sentences = len(SENTENCE_END.findall(spoken)) or 1
        rows.append((number, text, raw, spoken, puns, sentences))

    print(f"model: {model}\n")
    for number, text, raw, spoken, puns, sentences in rows:
        print(f"[{number}] user:   {text}")
        print(f"    jasper: {spoken}")
        if raw != spoken:
            print(f"    raw:    {raw}")
        print(f"    puns={puns or '-'} sentences={sentences}\n")

    detailed = len(TURNS) - 2  # the heat pump turn asks for detail
    with_pun = [r for r in rows if r[4]]
    used = [pun for r in rows for pun in r[4]]
    failures = {
        "self-descriptions": [r[0] for r in rows if SELF_DESCRIPTION.search(r[3]) and r[0] != 5],
        "over 3 sentences": [r[0] for r in rows if r[5] > 3 and r[0] != detailed],
        "more than one pun": [r[0] for r in rows if len(r[4]) > 1],
        "pun on serious turn": [r[0] for r in rows if r[4] and r[0] in SERIOUS],
        "repeated puns": sorted({pun for pun in used if used.count(pun) > 1}),
        "cleaned for TTS": [r[0] for r in rows if r[2] != r[3]],
        "empty replies": [r[0] for r in rows if not r[3]],
    }
    print(f"replies with a pun: {len(with_pun)}/{len(rows)} (target about 1 in 3)")
    for name, hits in failures.items():
        print(f"{name}: {hits or 'none'}")
    # "cleaned" is informational: the cleaner did its job, the prompt could be tighter.
    hard = {name: hits for name, hits in failures.items() if name != "cleaned for TTS"}
    pun_rate_ok = 1 <= len(with_pun) <= 5
    return 0 if pun_rate_ok and not any(hard.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
