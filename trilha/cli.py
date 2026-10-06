"""trilha — identify birds by ear on the trail, with no signal."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from . import __version__, config, journal, narrate
from .detect import Listener, week_of

LANG_TO_PROSE = {
    "en_us": "English",
    "pt": "Brazilian Portuguese",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
}


def _parse_date(raw: str) -> date:
    try:
        return date.fromisoformat(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {raw!r}") from None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trilha",
        description="Identify birds by ear on the trail, with no signal.",
    )
    parser.add_argument("--version", action="version", version=f"trilha {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    listen = sub.add_parser("listen", help="identify birds in one or more recordings")
    listen.add_argument("audio", nargs="+", type=Path)
    listen.add_argument("--lat", type=float, required=True, help="latitude, decimal degrees")
    listen.add_argument("--lon", type=float, required=True, help="longitude, decimal degrees")
    listen.add_argument("--date", type=_parse_date, default=None, help="YYYY-MM-DD (default: today)")
    listen.add_argument("--place", default=None, help="a name for this spot, for the journal")
    listen.add_argument("--min-conf", type=float, default=0.25, help="confidence floor (default: 0.25)")
    listen.add_argument("--top-k", type=int, default=3, help="species per 3s window (default: 3)")
    listen.add_argument("--lang", default="en_us", help="common-name language (default: en_us)")
    listen.add_argument("--workers", type=int, default=4, help="inference threads (default: 4)")
    listen.add_argument("--model", default=None, help="Ollama model for the field note")
    listen.add_argument(
        "--no-range",
        action="store_true",
        help="skip the range filter and listen for every species the model knows",
    )
    listen.add_argument("--no-note", action="store_true", help="skip the local field note")
    listen.add_argument("--no-journal", action="store_true", help="don't write to the journal")

    here = sub.add_parser("here", help="list the species in range at a coordinate")
    here.add_argument("--lat", type=float, required=True)
    here.add_argument("--lon", type=float, required=True)
    here.add_argument("--date", type=_parse_date, default=None)
    here.add_argument("--lang", default="en_us")
    here.add_argument("--limit", type=int, default=25, help="how many to print (0 = all)")

    show = sub.add_parser("journal", help="read or export the local field journal")
    show.add_argument("--markdown", action="store_true", help="print the journal as Markdown")
    show.add_argument("--export", type=Path, default=None, help="write the Markdown to a file")

    sub.add_parser("doctor", help="check models, Ollama, and local paths")
    return parser


def _print_listening(result, note: str | None) -> None:
    ranked = result.by_species()
    print()
    print(f"  {result.audio_path.name}")
    print(
        f"  {result.candidate_species} species "
        f"{'in range' if result.ranged else 'worldwide'} · "
        f"{result.elapsed_s:.1f}s on CPU · week {result.week}"
    )
    print()
    if not ranked:
        print("  Nothing above the confidence floor. Try --min-conf 0.1.")
    for det, windows in ranked:
        bar = "#" * max(1, round(det.confidence * 20))
        print(f"  {det.confidence:5.2f} {bar:<20} {det.common_name}")
        print(f"        {det.scientific_name} · first at {det.start_s:.0f}s · {windows} window(s)")
    if note:
        print()
        print("  Field note")
        for line in note.splitlines():
            print(f"  {line}" if line.strip() else "")
    print()


def _cmd_listen(args: argparse.Namespace) -> int:
    missing = [path for path in args.audio if not path.is_file()]
    if missing:
        for path in missing:
            print(f"trilha: no such file: {path}", file=sys.stderr)
        return 1

    listener = Listener(lang=args.lang)
    language = LANG_TO_PROSE.get(args.lang, "English")
    exit_code = 0

    for path in args.audio:
        try:
            result = listener.listen(
                path,
                latitude=args.lat,
                longitude=args.lon,
                when=args.date,
                min_confidence=args.min_conf,
                top_k=args.top_k,
                workers=args.workers,
                use_range=not args.no_range,
            )
        except RuntimeError as exc:
            print(f"trilha: {path.name}: {exc}", file=sys.stderr)
            exit_code = 1
            continue

        note = None
        if not args.no_note:
            try:
                note = narrate.field_note(
                    result, place=args.place, model=args.model, language=language
                )
            except narrate.NarrationUnavailable as exc:
                print(f"trilha: field note skipped — {exc}", file=sys.stderr)

        _print_listening(result, note)

        if not args.no_journal:
            entry = journal.entry_from(result, place=args.place, note=note)
            print(f"  saved to {journal.append(entry)}")
            print()

    return exit_code


def _cmd_here(args: argparse.Namespace) -> int:
    listener = Listener(lang=args.lang)
    week = week_of(args.date or date.today())
    species = listener.plausible_species(args.lat, args.lon, week)
    print()
    print(f"  {len(species)} species in range at {args.lat:.4f}, {args.lon:.4f}, week {week}")
    print()
    shown = species if args.limit == 0 else species[: args.limit]
    for label in shown:
        scientific, _, common = label.partition("_")
        print(f"  {common or scientific}  ({scientific})")
    if len(shown) < len(species):
        print(f"  ... and {len(species) - len(shown)} more (--limit 0 for all)")
    print()
    return 0


def _cmd_journal(args: argparse.Namespace) -> int:
    entries = journal.read_all()
    if args.export:
        args.export.write_text(journal.to_markdown(entries), encoding="utf-8")
        print(f"wrote {args.export}")
        return 0
    if args.markdown:
        print(journal.to_markdown(entries))
        return 0

    if not entries:
        print(f"Journal is empty ({config.journal_path()}).")
        return 0
    print()
    for entry in entries:
        where = entry["place"] or f"{entry['latitude']:.4f}, {entry['longitude']:.4f}"
        top = ", ".join(s["common_name"] for s in entry["species"][:3]) or "nothing"
        print(f"  {entry['recorded_at']}  {where}")
        print(f"      {top}")
    print()
    print(f"  {len(entries)} outing(s) in {config.journal_path()}")
    print()
    return 0


def _cmd_doctor(_: argparse.Namespace) -> int:
    print()
    print(f"  trilha {__version__}")
    print(f"  state      {config.home()}")
    print(f"  journal    {config.journal_path()} "
          f"({len(journal.read_all())} outing(s))")

    try:
        import birdnet  # noqa: F401
    except ImportError:
        print("  birdnet    NOT INSTALLED — pip install -r requirements.txt")
        return 1

    listener = Listener()
    try:
        acoustic = listener.acoustic
        print(f"  ear model  BirdNET {acoustic.get_version()} "
              f"({acoustic.n_species} species) {acoustic.model_path}")
        geo = listener.geo
        print(f"  range      BirdNET {geo.get_version()} ({geo.n_species} species)")
    except Exception as exc:  # the loader raises many kinds on a cold cache
        print(f"  models     NOT READY — {exc}")
        print("  Run `trilha here --lat 0 --lon 0` once online to download them.")
        return 1

    ok, detail = narrate.available()
    print(f"  field note {'ready' if ok else 'unavailable'} — {detail}")
    print()
    print("  Models are cached on disk. After this check you can go offline.")
    print()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    handlers = {
        "listen": _cmd_listen,
        "here": _cmd_here,
        "journal": _cmd_journal,
        "doctor": _cmd_doctor,
    }
    try:
        return handlers[args.command](args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
