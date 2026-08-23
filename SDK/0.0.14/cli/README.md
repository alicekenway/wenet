# SDK 0.0.14 command-line tools

## `asr_package_eval`

This tool compares the package decoder (`lm`) with CTC greedy decoding on a
JSONL test set. In LM debug mode, each output row also contains structured
`decoder_debug` data, including the winning fixed LM and its raw, UNK-reference,
adjusted, and weighted score for every completed output event.

```bash
build_008/asr_package_eval \
  --model_dir /path/to/package \
  --metadata /path/to/metadata.jsonl \
  --wav_parent /path/to/dataset \
  --output_json /path/to/output.jsonl \
  --decode_mode lm \
  --debug true \
  --debug_log /path/to/output.debug.txt
```

Important options:

```text
--decode_mode lm|greedy   Select max-fusion decoding or AM-only greedy CTC.
--limit N                 Decode only the first N input rows.
--num_threads N           Set inference thread count.
--chunk_ms N              Feed streaming chunks; zero feeds the whole WAV.
--slots PATH              Load generic runtime slot values (LM mode only).
--contact_names PATH      Compatibility input for <CONTACT> values.
--contacts_tsv PATH       Compatibility input with explicit contact IDs/forms.
--debug true|false        Include structured decoder_debug in LM JSONL rows.
--debug_log PATH          Also write compact ref/hyp/AM/LM N-best blocks.
```

`--slots` uses sections headed by a configured slot token:

```text
#<CONTACT>
Alice
Bob Smith

#<ADDRESS>
Beijing Station
```

Only one of `--slots`, `--contact_names`, and `--contacts_tsv` may be used.
The package must contain a bias LM configured for every supplied slot class.

Input JSONL defaults to `text`, `sentence`, or `transcript` for the reference,
and `audiofile_path`, `audio_filepath`, `wav`, `path`, or `file_name` for audio.
Relative WAV paths are resolved below `--wav_parent`. Output rows retain the
input fields and add `hyp`, `rtf`, `atf`, `decode_sec`, `audio_sec`, and
`decode_mode`; failed rows also contain `error`.

## `asr_stream_file`

`asr_stream_file` decodes one WAV and supports the same `--slots` section
file. Existing `--contacts_tsv` input remains available and maps to the
`<CONTACT>` slot for source compatibility.

## `summarize_asr_package_eval.py`

The summarizer computes WER/CER, SER, aggregate RTF, and optional per-utterance
details from evaluator JSONL output:

```bash
python3 cli/summarize_asr_package_eval.py \
  --input_json /path/to/output.jsonl \
  --summary /path/to/summary.txt \
  --detail_json /path/to/detail.jsonl
```

Use `--metric wer`, `--metric cer`, or the default `--metric auto`. Optional
`--ref_regex_rules` applies tab-separated regex replacements to reference text
before scoring.
