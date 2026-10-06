# trilha

Identify birds by ear on the trail, with no signal.

You hear something in the brush. You don't know what it is. Normally that means
holding your phone up to the canopy and hoping for a bar of signal, or getting
home six hours later to a recording you can no longer place.

`trilha` does the whole thing on your laptop, offline: it listens to a
recording, names what it heard, and writes the outing into a field journal in
your own words — all from open weights running locally.

```
$ trilha listen manha.wav --lat -23.55 --lon -46.63 --place "Parque do Carmo"

  manha.wav
  596 species in range · 5.2s on CPU · week 40

   0.97 ###################  Rufous-bellied Thrush
        Turdus rufiventris · first at 30s · 22 window(s)
   0.94 ###################  Rufous-collared Sparrow
        Zonotrichia capensis · first at 42s · 26 window(s)
   0.92 ##################   Southern Lapwing
        Vanellus chilensis · first at 63s · 4 window(s)
   0.59 ############         Southern Yellowthroat
        Geothlypis velata · first at 78s · 2 window(s)

  Field note
  The presence of the Rufous-bellied Thrush, Rufous-collared Sparrow, and
  Southern Lapwing strongly suggests a mixed habitat of open woodlands and
  grasslands within Parque do Carmo. Next, while on this trail, pay particular
  attention to the calls of the Southern Yellowthroat — they often favour dense
  undergrowth and shrubbery, and their distinctive song is relatively
  high-pitched.
```

No account, no API key, no upload. Your coordinates never leave the machine.

## How it works

Two open models, each doing the thing it is good at:

**The ear** — [BirdNET](https://github.com/birdnet-team/birdnet) 3.0, running
through ONNX on CPU. It knows 11,560 species and slices the audio into 3-second
windows, scoring each one.

**The filter** — BirdNET's range model (14,082 species) takes your latitude,
longitude and week of the year and narrows the candidates to what could
plausibly be there. At `-23.55, -46.63` in week 40 that is 596 species instead
of 11,560, which is the difference between a useful answer and a list of
everything that has ever had feathers.

**The voice** — [Gemma 3](https://ai.google.dev/gemma) through
[Ollama](https://ollama.com), on your machine, turning the detection list into a
few sentences of prose for the journal. It is told never to name a species that
was not detected, and never to guess a place name from bare coordinates. If
Ollama isn't running, the note is skipped and the identifications still stand —
**nothing about the identification depends on the language model.**

## Install

Needs Python 3.11+.

```bash
git clone https://github.com/Tinhomagri/trilha-offline
cd trilha-offline
python3 -m venv .venv
.venv/bin/pip install -e .
```

For the field notes, install [Ollama](https://ollama.com) and pull a model:

```bash
ollama pull gemma3:4b
```

Then check everything is cached and ready:

```bash
$ trilha doctor

  trilha 0.1.0
  state      ~/.local/share/trilha
  journal    ~/.local/share/trilha/journal.jsonl (0 outing(s))
  ear model  BirdNET 3.0 (11560 species)
  range      BirdNET 3.0 (14082 species)
  field note ready — gemma3:4b ready

  Models are cached on disk. After this check you can go offline.
```

Run `doctor` once at home, on wifi. After that the models live on disk and the
whole tool works with the radio off.

## Commands

### `listen` — identify birds in one or more recordings

```bash
trilha listen dawn.wav dusk.wav --lat -23.55 --lon -46.63
```

| Flag | Default | |
|---|---|---|
| `--lat`, `--lon` | *required* | decimal degrees, for the range filter |
| `--date` | today | `YYYY-MM-DD`, if you're processing an older recording |
| `--place` | — | a name for this spot, for the journal |
| `--min-conf` | `0.25` | confidence floor |
| `--top-k` | `3` | species per 3s window |
| `--lang` | `en_us` | common-name language (`pt`, `es`, `fr`, `de`, …) |
| `--workers` | `4` | inference threads |
| `--model` | `gemma3:4b` | Ollama model for the field note |
| `--no-note` | | skip the field note |
| `--no-journal` | | don't record the outing |

### `here` — what's in range at a coordinate

No audio needed. Useful the night before, to know what you might hear.

```bash
trilha here --lat -23.55 --lon -46.63 --limit 25
```

### `journal` — read or export the field journal

Every outing is appended to `~/.local/share/trilha/journal.jsonl`.

```bash
trilha journal                          # read it
trilha journal --markdown               # as Markdown
trilha journal --export trilha.md       # write it out
```

### `doctor` — check models, Ollama and local paths

## Configuration

| Variable | Default |
|---|---|
| `TRILHA_HOME` | `~/.local/share/trilha` |
| `TRILHA_OLLAMA_URL` | `http://localhost:11434` |
| `TRILHA_OLLAMA_MODEL` | `gemma3:4b` |

## Why open weights, specifically

This tool does not work without them, and that is the whole point:

- **No signal is the normal case.** The places worth birding are the places
  with no bars. A closed API is a hard dependency on a network that isn't
  there. Every model here is a file on disk.
- **Coordinates are sensitive.** Where you go birding, at what hour, how often
  — that is a movement profile. Sending it to someone else's server to get a
  bird's name back is a bad trade. Here it never leaves the machine.
- **The range filter needs the geography, not a guess.** Narrowing 11,560
  species to 596 by latitude and week is what makes the identification
  trustworthy, and it runs locally on a model you can inspect.
- **Swappable voice.** `--model` takes any model Ollama can run. The note is
  the one part with taste in it, so it should be the part you can change —
  from `gemma3:4b` on a laptop to `gemma3:12b` when you have the RAM.
- **Costs nothing per use.** A birder can run this every weekend for a decade.

## Samples

Two recordings from [xeno-canto](https://xeno-canto.org) are included for
trying it out without going outside first:

```bash
trilha listen samples/corruira-XC436932.mp3 --lat -23.55 --lon -46.63 --lang pt
```

## License

MIT. BirdNET and Gemma 3 carry their own licenses.
