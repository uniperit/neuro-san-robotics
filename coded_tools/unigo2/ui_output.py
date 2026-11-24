"""Agent-owned presentation tool for the robot's browser and speaker."""

from __future__ import annotations

import asyncio
import re
from typing import Any
from typing import Dict

from neuro_san.interfaces.coded_tool import CodedTool

from coded_tools.unigo2.agent_events import SILENT_SOURCES
from coded_tools.unigo2.agent_events import event_source
from coded_tools.unigo2.agent_events import publish_ui_output

_STAGE_DIRECTION = re.compile(r"\*[^*\n]+\*(?!\*)")  # *wags tail*, not **bold**
_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_LIST_MARKER = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+", re.MULTILINE)
_MARKUP = re.compile(r"[*_#`~>|]+")
# Emoji, pictographs, dingbats, flags, variation selectors and joiners.
_EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]")


def speakable(text: str) -> str:
    """Drop what TTS would read out literally: stage directions, markdown, emojis."""
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)  # bold keeps its words
    text = _STAGE_DIRECTION.sub(" ", text)
    text = _MARKDOWN_LINK.sub(r"\1", text)
    text = _LIST_MARKER.sub("", text)
    text = _EMOJI.sub("", _MARKUP.sub("", text))
    text = re.sub(r"\s+([,.!?])", r"\1", text)
    return " ".join(text.split())


class UiOutputTool(CodedTool):
    """Publish only explicit agent thoughts and speech to the presentation layer."""

    async def async_invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]) -> Any:
        thought = args.get("thought", "")
        say = args.get("say", "")
        heard = args.get("heard", "")
        if (
            not isinstance(thought, str)
            or not isinstance(say, str)
            or not isinstance(heard, str)
        ):
            return "thought, say, and heard must be strings. End this event turn now."
        say = speakable(say)
        if event_source(sly_data) in SILENT_SOURCES:
            say = ""
        if not thought.strip() and not say.strip():
            return (
                "Silence selected. End this event turn now without calling "
                "ui_output or any other tool again."
            )

        delivered = await asyncio.to_thread(
            publish_ui_output,
            thought=thought,
            say=say,
            heard=heard,
            # The epoch the dispatching turn was sent with; lets the UI drop
            # this speech if the user has since barged in.
            epoch=(sly_data or {}).get("speech_epoch"),
            # Ties this output to the voice turn's timing line (#13).
            turn_id=(sly_data or {}).get("turn_id"),
        )
        if delivered:
            return "Output delivered. End this event turn now; do not call ui_output again."
        return (
            "The presentation adapter is unavailable. End this event turn now; "
            "do not retry ui_output."
        )
