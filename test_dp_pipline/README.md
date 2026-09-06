# Shared DP test pipeline

Python 3.10+; standard library only. Uses each SDK's existing evaluator and scorer,
without modifying SDK code, generating contact paths, or converting packages.

```bash
python3 wenet/test_dp_pipline/run.py \
  --sdk-dir /home/jinyang_wang/Dev/ASR/ASR_wenet/wenet/SDK \
  --version 0.0.16 \
  --test-set /path/to/test_set.json \
  --mode lm --itn off --ref-sub /path/to/ref_text_normalization_rules.tsv
```

`--sdk-dir` accepts either the SDK collection root or a version's source directory.
Supported versions: **0.0.13, 0.0.14, 0.0.15, 0.0.16**. Build that SDK's Linux host
`asr_package_eval` first. The runner checks the CMake source version and searches
`build`, `build_0_0_XX`, and `build_00XX`. It does not select Android/ASan builds
automatically. Use `--evaluator /path/to/asr_package_eval` for a custom build or
to disambiguate multiple builds. Source-version validation does not prove the
provenance of an externally supplied binary; supply the matching SDK build.

## Dataset settings

Store paths once in a small JSON file. Relative paths resolve against that file,
not the working directory. Example:

```json
{
  "metadata": "manifest.jsonl",
  "wav_parent": ".",
  "package": "/path/to/package",
  "contacts": "name_test.txt",
  "ref_rules": "ref_text_normalization_rules.tsv",
  "output_dir": "/path/to/results"
}
```

There is one package path, with no alternate-package selection or fallback.
In LM mode, versions 13–15 use the text lexicon and do not require `lexicon.bin`.
Version 16 requires `lexicon.bin`; a missing file is an error, not a request to
generate it. The manifest must also use the SDK-compatible Flashlight decoder
type; version 16 requires `compact_trie_v1` and a packaged SentencePiece model
when contacts are enabled. The runner does not rewrite the manifest. Greedy mode
only needs AM/token inputs and does not require a lexicon binary for any version.

Manifest records use `audio_filepath`, `text`, and positive finite `duration`
(seconds). Audio paths may be absolute or relative to `wav_parent`, which defaults
to the manifest directory. All recordings are checked before any results move.
This runner targets the existing SDK JSONL/CTC evaluation workflow, not HRL/CRP
conversion or arbitrary decoder backends.

Omit `contacts` (or use null) to disable contact injection. Contacts are plain names:
the SDK generates their spellings. **Strict mode is always enabled**; older SDKs
may still reject complex names due to their existing path limit. The runner
reports that failure rather than skipping names or supplying a manual workaround.

Optional settings: `mode` (lm/greedy/both, default lm), `metric` (wer/cer/auto,
default wer), `threads` (default 1), `itn` and `debug` (booleans, default false).
`ref_rules` is optional. Unknown fields and duplicate JSON keys are rejected.

## Check or run

- `--check`: validate inputs and print resolved commands; no result writes/backups.
- `--limit 2`: real two-record smoke test, scored normally.
- `--mode both`: override the configured mode.
- `--itn on` / `--itn off`: enable/disable ITN, overriding the JSON setting.
  Enabling ITN requires the package's English tagger/verbalizer FSTs.
- `--ref-sub FILE`: reference substitutions applied by the selected SDK's scorer;
  overrides JSON `ref_rules`. Aliases: `--ref-sub-file`, `--ref-rules`.
  Rules use the scorer's existing format: regex and replacement separated by
  a tab or ` -> `; a regex alone removes matches. This changes references used
  for scoring, not recognized hypotheses or package mappings.
- `--output-dir /tmp/my-test`: override output location for a smoke test.

Command-line overrides take precedence over dataset settings. Command-line file
paths are relative to the current working directory; JSON paths remain relative
to the settings file. Omit `ref_rules` from JSON and omit `--ref-sub` to score
without reference substitutions.

Results go to `OUTPUT/sdk_VERSION/work_lm` (or `work_greedy`): hypotheses,
per-record details, summary, evaluation/scoring logs, and `run_info.json` with
resolved settings, commands, and completion/failure status. Debug output is
written only when enabled. Previous mode results move to a uniquely named folder
under `OUTPUT/bak`; original inputs and other SDK results remain untouched.
A per-version lock prevents concurrent writers. A crash may leave a lock;
check the recorded PID and remove it only after confirming the runner has stopped.

The example ENX dataset script now only supplies the SDK location/version and
`test_set.json`. To switch SDKs, change `SDK_VERSION` or pass `--version 0.0.14`.
Its single package path references `DP/4wheels/ENX_26_09_05_4wheels`.
Ensure that directory contains the files and manifest required by the selected
SDK; archived packages are not used as fallbacks.

Run automated runner tests with:

```bash
python3 -m unittest discover -s wenet/test_dp_pipline -p 'test_*.py' -v
```

These fixture tests verify runner compatibility/contracts for all four versions;
they do not replace building and validating the native SDKs themselves.
