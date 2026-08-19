# English inverse text normalization

This directory owns the product rules layered on WeTextProcessing v1.2.0.
The upstream source is vendored in `SDK/0.0.9/third_party/WeTextProcessing`.

```bash
python -m ITN.export --language en
python -m ITN.test --language en
python -m ITN.test --text "F M one hundred point nine"
```

Generated `english/export/en_itn_{tagger,verbalizer}.fst` files are release
assets and are intentionally small enough to keep with the grammar sources.

## Adding a rule

Keep product rules under `ITN/english/rules/` and their editable lookup data
under `ITN/english/data/<rule-name>/`. A rule has two transducers:

- the **tagger** recognizes spoken text and produces a named token;
- the **verbalizer** removes that token wrapper and emits product text.

For example, `rules/radio.py` reads `data/radio/band.tsv` and builds this
conceptual transformation:

```text
f m one hundred point nine
  -> radio { value: "FM100.9" }
  -> FM100.9
```

To add a new rule:

1. Copy the shape of `rules/radio.py`; choose a unique token name and keep the
   tagger and verbalizer field order identical.
2. Import and instantiate the rule in `english/inverse_normalizer.py`.
3. Add its tagger near the front of `classify` with a low weight so it wins
   over generic word/cardinal paths.
4. Add its verbalizer to the `verbalizer` union.
5. Add positive and conservative negative examples to `ITN/test.py`.

The runtime lowercases input before composition, so rule inputs should be
lowercase. Outputs may use canonical case such as `FM` or `MH`.

## Exporting and testing FSTs

Install WeTextProcessing's development dependencies in an isolated Python
environment (`pynini>=2.1.6` and `importlib_resources`), then run from the
repository root:

```bash
python -m ITN.export --language en --output-dir ITN/english/export
python -m ITN.test --language en
python -m ITN.test --language en --text "M H three seventy"
```

`export` always rebuilds both files. `test` loads those exact on-disk FSTs, so
it also checks that a release can read the exported assets.

To create a reference manifest whose `text` matches ITN-enabled hypotheses:

```bash
python -m ITN.test \
  --input-jsonl ../data/ENX/ENX_batch1_control/manifest.jsonl \
  --output-jsonl ../data/ENX/ENX_batch1_control/manifest.itn.jsonl

python -m ITN.test \
  --input-text ../data/ENX/ENX_batch1_control/metadata.txt \
  --output-text ../data/ENX/ENX_batch1_control/metadata.itn.txt
```

## Enabling ITN in an SDK package

ITN is off when the three manifest keys are absent, which keeps ordinary
WER/CER evaluation in spoken-text space. Enable it by adding all three keys:

```json
{
  "itn_language": "en",
  "itn_tagger": "en_itn_tagger.fst",
  "itn_verbalizer": "en_itn_verbalizer.fst"
}
```

For the Flashlight packaging workflow, set `ITN_LANGUAGE=en`, `ITN_TAGGER`,
and `ITN_VERBALIZER`; the script copies the files, adds the keys, and includes
the FSTs in `checksums.sha256`.

The SDK's normal product default is `EngineConfig.enable_itn = true`. The
evaluation CLI defaults to ITN off even when the package contains FSTs; opt in
with `--enable_itn true`. The dataset wrapper exposes the same choice as
`ENABLE_ITN=true`. This allows one ITN-capable package to be evaluated in
spoken form or product form without editing its manifest.
