"""Local bird-call detection: BirdNET 3.0 on CPU, narrowed by where you are.

The acoustic model knows 11,560 species worldwide. The range model knows which
of them actually occur at a coordinate in a given week. Intersecting the two is
what keeps a Brazilian trail from coming back full of North American warblers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from . import config


@dataclass(frozen=True)
class Detection:
    """One species heard in one 3-second window."""

    scientific_name: str
    common_name: str
    start_s: float
    end_s: float
    confidence: float

    @property
    def label(self) -> str:
        return f"{self.common_name} ({self.scientific_name})"


@dataclass
class Listening:
    """Everything one audio file yielded."""

    audio_path: Path
    latitude: float
    longitude: float
    week: int
    detections: list[Detection] = field(default_factory=list)
    candidate_species: int = 0
    elapsed_s: float = 0.0

    def by_species(self) -> list[tuple[Detection, int]]:
        """Best detection per species plus how many windows it was heard in."""
        best: dict[str, Detection] = {}
        counts: dict[str, int] = {}
        for det in self.detections:
            key = det.scientific_name
            counts[key] = counts.get(key, 0) + 1
            if key not in best or det.confidence > best[key].confidence:
                best[key] = det
        ranked = sorted(best.values(), key=lambda d: d.confidence, reverse=True)
        return [(det, counts[det.scientific_name]) for det in ranked]


def week_of(when: date) -> int:
    """BirdNET's range model wants a week of year in 1..48."""
    return min(48, max(1, (when.timetuple().tm_yday - 1) // 7 + 1))


def _split_label(label: str) -> tuple[str, str]:
    scientific, _, common = label.partition("_")
    return scientific, common or scientific


class Listener:
    """Holds the two models open so repeated files don't reload them."""

    def __init__(self, lang: str = "en_us") -> None:
        self.lang = lang
        self._acoustic = None
        self._geo = None
        self._geo_cache = self._read_geo_cache()

    # -- models ------------------------------------------------------------

    @property
    def acoustic(self):
        if self._acoustic is None:
            import birdnet

            self._acoustic = birdnet.load(
                "acoustic", config.ACOUSTIC_VERSION, config.BACKEND,
                precision="fp32", lang=self.lang,
            )
        return self._acoustic

    @property
    def geo(self):
        if self._geo is None:
            import birdnet

            self._geo = birdnet.load(
                "geo", config.GEO_VERSION, config.BACKEND, lang=self.lang,
            )
        return self._geo

    # -- range filter ------------------------------------------------------

    def _read_geo_cache(self) -> dict[str, list[str]]:
        path = config.geo_cache_path()
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _write_geo_cache(self) -> None:
        try:
            config.geo_cache_path().write_text(
                json.dumps(self._geo_cache), encoding="utf-8"
            )
        except OSError:
            pass

    def plausible_species(
        self, latitude: float, longitude: float, week: int, min_confidence: float = 0.03
    ) -> list[str]:
        """Species the range model expects here, that the ear model can also name."""
        key = f"{self.lang}|{latitude:.3f}|{longitude:.3f}|{week}|{min_confidence}"
        if key in self._geo_cache:
            return self._geo_cache[key]

        expected = self.geo.predict(
            latitude, longitude, week=week, min_confidence=min_confidence
        ).to_set()
        audible = set(self.acoustic.species_list)
        species = sorted(expected & audible)

        self._geo_cache[key] = species
        self._write_geo_cache()
        return species

    # -- inference ---------------------------------------------------------

    def listen(
        self,
        audio_path: Path,
        latitude: float,
        longitude: float,
        when: date | None = None,
        min_confidence: float = 0.25,
        top_k: int = 3,
        workers: int = 4,
    ) -> Listening:
        import time

        week = week_of(when or date.today())
        allowed = self.plausible_species(latitude, longitude, week)
        if not allowed:
            raise RuntimeError(
                "The range model expects no species at this coordinate. "
                "Check --lat/--lon, or pass --no-range to listen worldwide."
            )

        started = time.monotonic()
        result = self.acoustic.predict(
            audio_path,
            top_k=top_k,
            custom_species_list=allowed,
            default_confidence_threshold=min_confidence,
            n_workers=workers,
            batch_size=8,
        )
        elapsed = time.monotonic() - started

        detections = []
        for row in result.to_dataframe().itertuples():
            scientific, common = _split_label(row.species_name)
            detections.append(
                Detection(
                    scientific_name=scientific,
                    common_name=common,
                    start_s=float(row.start_time),
                    end_s=float(row.end_time),
                    confidence=float(row.confidence),
                )
            )

        return Listening(
            audio_path=audio_path,
            latitude=latitude,
            longitude=longitude,
            week=week,
            detections=detections,
            candidate_species=len(allowed),
            elapsed_s=elapsed,
        )
