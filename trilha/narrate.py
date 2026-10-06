"""Turn a detection list into a field note, using a local open-weight model.

Runs Gemma 3 through Ollama on this machine. If Ollama isn't up, the note is
skipped and the detections still stand on their own — the identification never
depends on the language model.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from . import config
from .detect import Listening

SYSTEM = """You are a field naturalist writing a note in a birder's paper journal.
You are given bird species a recorder identified at a location, with confidence
scores. Write 3 to 5 sentences. Say what the mix of species suggests about the
habitat, and name one concrete thing the listener should look or listen for next
on this trail. Be specific and plain. Never invent a species that is not listed,
and never invent a count. If confidence is low, say so.

Only ever refer to the location exactly as it is given to you. When you are given
coordinates and no place name, leave the place unnamed — never guess a park,
reserve, region, biome or city from them.

Return only the note itself: no preamble, no sign-off, no heading, no title, no
Markdown formatting, no restating of this instruction. Start with the first
sentence of the note."""


class NarrationUnavailable(RuntimeError):
    """Ollama isn't reachable, or the model isn't pulled."""


def _prompt(listening: Listening, place: str | None) -> str:
    lines = [
        f"Location: {place or f'{listening.latitude:.4f}, {listening.longitude:.4f}'}",
        f"Week of year: {listening.week}",
        f"Recording: {listening.audio_path.name}",
        "Species identified (best confidence, number of 3s windows):",
    ]
    for det, windows in listening.by_species():
        lines.append(f"- {det.label}: {det.confidence:.2f} in {windows} window(s)")
    lines.append("")
    lines.append("Write the field note. Output the note text and nothing else.")
    return "\n".join(lines)


def field_note(
    listening: Listening,
    place: str | None = None,
    model: str | None = None,
    language: str = "English",
    timeout_s: float = 180.0,
) -> str:
    if not listening.detections:
        raise NarrationUnavailable("nothing was detected, so there is nothing to narrate")

    payload = {
        "model": model or config.OLLAMA_MODEL,
        "system": f"{SYSTEM}\n\nWrite the note in {language}.",
        "prompt": _prompt(listening, place),
        "stream": False,
        "options": {"temperature": 0.4, "num_predict": 320},
    }
    request = urllib.request.Request(
        f"{config.OLLAMA_URL}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise NarrationUnavailable(
            f"Ollama rejected the request ({exc.code}). "
            f"Is '{payload['model']}' pulled? Try: ollama pull {payload['model']}"
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise NarrationUnavailable(
            f"No local model at {config.OLLAMA_URL}. Start it with: ollama serve"
        ) from exc

    note = (body.get("response") or "").strip()
    if not note:
        raise NarrationUnavailable("the local model returned an empty note")
    return note


def available(model: str | None = None) -> tuple[bool, str]:
    """Check Ollama and the model without generating anything."""
    wanted = model or config.OLLAMA_MODEL
    try:
        with urllib.request.urlopen(f"{config.OLLAMA_URL}/api/tags", timeout=5) as response:
            tags = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return False, f"Ollama not reachable at {config.OLLAMA_URL} ({exc})"

    names = [entry.get("name", "") for entry in tags.get("models", [])]
    if wanted in names:
        return True, f"{wanted} ready"
    return False, f"{wanted} not pulled. Available: {', '.join(names) or 'none'}"
