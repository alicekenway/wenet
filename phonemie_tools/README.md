# Phoneme manifest tools

`nemo_jsonl_to_phonemes.py` converts the transcript field in a UTF-8 JSONL
manifest to space-separated phoneme tokens. It preserves record order and all
other JSON fields.

## English and other eSpeak languages

The existing eSpeak backend remains the default:

```bash
python phonemie_tools/nemo_jsonl_to_phonemes.py \
  input.jsonl output.jsonl \
  --backend espeak \
  --language en-us \
  --workers 8
```

`--word-token` controls the eSpeak word-boundary token and defaults to `▁`.
Use `--word-token none` to disable it.

## Mandarin G2PW IPA

Install the pinned IPA-capable `g2p-mix` revision and its G2PW dependencies:

```bash
python -m pip install -r phonemie_tools/requirements-g2pw.txt
```

Then convert Mandarin or mixed Mandarin-English transcripts:

```bash
python phonemie_tools/nemo_jsonl_to_phonemes.py \
  input.jsonl output.jsonl \
  --backend g2pw \
  --workers 1
```

The G2PW path writes `g2p-mix`'s IPA `result.phones` directly, separated by
spaces. Mandarin tone contours are retained and tone sandhi is enabled. It
does not add a word-boundary token. Unknown Chinese characters are strict: an
unsupported pronunciation stops conversion instead of silently dropping a
training label.

Reviewed Mainland Mandarin corrections live in the independent mapping file
`g2pw_pinyin_overrides.json`. It currently fixes an incompatibility in the
downloaded G2PW model: its `崖` label `yai2` (Taiwan-style `ㄧㄞ2`) is normalized
to standard Hanyu Pinyin `ya2` (`ㄧㄚ2`) before IPA conversion. Each override is
nested under its exact character and original syllable, so `yai2` is not
replaced globally. For example:

```json
{
  "崖": {
    "yai2": "ya2"
  }
}
```

The first non-empty G2PW conversion downloads the `pengzhendong/g2pw` model
snapshot through ModelScope unless it is already cached. Pre-cache it before
running in an offline environment. Each worker loads its own model, so the
default for `--backend g2pw` is one worker; increasing `--workers` increases
memory use. When more than one G2PW worker is requested, the converter first
performs one mixed Mandarin-English warm-up conversion in an isolated process.
This serializes the initial ModelScope and g2p-en/NLTK cache setup before the
worker processes are started, without forking a live ONNX Runtime session.

If a virtual environment was created from Miniforge/Conda and a cluster's
compute nodes have an older system C++ runtime, make the environment's library
directory visible when launching Python. A missing `GLIBCXX` symbol can be
wrapped by `g2p-mix` as the misleading error `English homograph rules are
unavailable`:

```bash
srun env LD_LIBRARY_PATH=/path/to/miniforge3/lib:$LD_LIBRARY_PATH \
  /path/to/g2pw_env/bin/python \
  phonemie_tools/nemo_jsonl_to_phonemes.py \
  input.jsonl output.jsonl --backend g2pw --workers 1
```

By default the JSON field named `text` is replaced. Use `--text-key` for a
different transcript field. Empty transcripts remain empty, blank physical
lines are skipped, and all non-transcript values are preserved.

## Progress and ETA

The converter counts non-blank JSONL records before inference and reports:

- completed and total records;
- completion percentage;
- elapsed wall-clock time;
- average records per second;
- approximate remaining time.

While model initialization or a long batch is running, it prints a heartbeat
every 30 seconds. `--progress-interval` changes that time interval, and
`--progress-every` also triggers a report after a chosen number of completed
records. Set both options to `0` to disable progress output:

```bash
python phonemie_tools/nemo_jsonl_to_phonemes.py \
  input.jsonl output.jsonl \
  --backend g2pw \
  --progress-interval 30 \
  --progress-every 1000
```

ETA is unavailable until the first batch completes. It is an estimate based on
average throughput since startup, so it becomes more stable as additional
batches finish.
