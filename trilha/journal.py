"""An append-only field journal on your own disk. Nothing is uploaded."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .detect import Listening


@dataclass(frozen=True)
class Entry:
    recorded_at: str
    audio: str
    latitude: float
    longitude: float
    week: int
    place: str | None
    candidate_species: int
    elapsed_s: float
    species: list[dict]
    note: str | None

    def as_dict(self) -> dict:
        return {
            "recorded_at": self.recorded_at,
            "audio": self.audio,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "week": self.week,
            "place": self.place,
            "candidate_species": self.candidate_species,
            "elapsed_s": round(self.elapsed_s, 2),
            "species": self.species,
            "note": self.note,
        }


def entry_from(listening: Listening, place: str | None, note: str | None) -> Entry:
    species = [
        {
            "scientific_name": det.scientific_name,
            "common_name": det.common_name,
            "confidence": round(det.confidence, 4),
            "windows": windows,
            "first_heard_s": det.start_s,
        }
        for det, windows in listening.by_species()
    ]
    return Entry(
        recorded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        audio=str(listening.audio_path),
        latitude=listening.latitude,
        longitude=listening.longitude,
        week=listening.week,
        place=place,
        candidate_species=listening.candidate_species,
        elapsed_s=listening.elapsed_s,
        species=species,
        note=note,
    )


def append(entry: Entry) -> Path:
    path = config.journal_path()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry.as_dict(), ensure_ascii=False) + "\n")
    return path


def read_all() -> list[dict]:
    path = config.journal_path()
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def to_markdown(entries: list[dict]) -> str:
    if not entries:
        return "# Field journal\n\nNothing recorded yet.\n"

    seen: dict[str, dict] = {}
    for entry in entries:
        for species in entry["species"]:
            key = species["scientific_name"]
            current = seen.get(key)
            if current is None or species["confidence"] > current["confidence"]:
                seen[key] = species

    out = ["# Field journal", ""]
    out.append(f"{len(entries)} recording(s), {len(seen)} species.")
    out.append("")
    out.append("## Life list")
    out.append("")
    for species in sorted(seen.values(), key=lambda s: s["common_name"]):
        out.append(
            f"- **{species['common_name']}** — *{species['scientific_name']}* "
            f"(best {species['confidence']:.2f})"
        )
    out.append("")
    out.append("## Outings")
    out.append("")
    for entry in entries:
        where = entry["place"] or f"{entry['latitude']:.4f}, {entry['longitude']:.4f}"
        out.append(f"### {entry['recorded_at']} — {where}")
        out.append("")
        out.append(f"`{Path(entry['audio']).name}` · week {entry['week']} · "
                   f"{entry['candidate_species']} species in range")
        out.append("")
        for species in entry["species"]:
            out.append(
                f"- {species['common_name']} ({species['confidence']:.2f}, "
                f"{species['windows']} window(s))"
            )
        if entry.get("note"):
            out.append("")
            out.append(f"> {entry['note']}")
        out.append("")
    return "\n".join(out)
