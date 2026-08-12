# KenLM/Flashlight Runtime Tools for SDK 0.0.3

This directory keeps the reusable data-processing commands for the SDK 0.0.3
Flashlight + KenLM path.

Run order:

```bash
cd /path/to/ASR_wenet

python3 wenet/kenlm_lm/tools/sample_eval_metadata.py
wenet/kenlm_lm/run/1_train_lm.sh
wenet/kenlm_lm/run/2_convert_to_bin.sh
wenet/kenlm_lm/run/3_generate_words_lm.sh
wenet/kenlm_lm/run/4_generate_lexicon_for_am.sh
wenet/kenlm_lm/run/6_prepare_runtime.sh
wenet/kenlm_lm/tools/run_003_acceptance_eval.sh
```

Default inputs:

- AM package: `model/sherpa-onnx-en-wenet-gigaspeech_int8`
- LM text: `LM/wenet_lm/training/preprocess_data/wenetspeech_lm_char.txt`
- Eval metadata: `data/hf_wenetspeech_test_net/wenetspeech_test_net_sample_2000/metadata.jsonl`

Main outputs:

- `wenet/kenlm_lm/models/wenetspeech_char_4gram.arpa`
- `wenet/kenlm_lm/models/lm.bin`
- `wenet/kenlm_lm/data/words.txt`
- `wenet/kenlm_lm/data/lexicon.txt`
- `test/0.0.3/model_flashlight`
- `test/0.0.3/acceptance`
- `test/0.0.3/acceptance/eval.tsv`, with `ref`, `greedy`, `lm`, and CER columns.
- `test/0.0.3/sdk_batch_flashlight_eval.tsv`, when scoring public SDK batch output.

The default KenLM pruning is `0 0 1 1`. Override it like this:

```bash
PRUNE="0 0 1 2" wenet/kenlm_lm/tools/train_kenlm_arpa.sh
```

If output is empty or strange, first check:

- `tokens.txt` IDs match the ONNX output dimension.
- `lexicon.txt` spelling tokens all exist in AM `tokens.txt`.
- `generate_lexicon_for_am.py --tokenization` matches the AM token style:
  use `byte` for byte-fallback character models and `bpe` for
  SentencePiece/BPE models with word-start tokens such as `▁THE`.
- Add `--ignore-case` when the LM words and AM tokens use different letter
  casing, for example uppercase LM words with lowercase AM BPE tokens.
- `words.txt` contains the same output units used to train KenLM.
- `output_mapping.txt` is empty or contains valid `source -> target` rules.
- `test/0.0.3/acceptance/*.log` for raw words, mapped words, scores, and RTF.
- `test/0.0.3/acceptance/summary.txt` for greedy-vs-LM aggregate CER.

For English BPE/SentencePiece AM tokens, build the lexicon like this:

```bash
PYTHON=/path/to/python-with-sentencepiece \
SENTENCEPIECE_MODEL=/path/to/the/am_training_tokenizer.model \
BPE_MAX_SPELLINGS=3 \
BPE_ADD_CHARACTER_FALLBACK=true \
wenet/kenlm_lm/run/4_generate_lexicon_for_am.sh
```

The equivalent direct command is:

```bash
python3 wenet/kenlm_lm/tools/generate_lexicon_for_am.py \
  --words wenet/kenlm_lm/data/words.txt \
  --tokens model/sherpa-onnx-en-wenet-gigaspeech_int8/tokens.txt \
  --output wenet/kenlm_lm/data/lexicon.txt \
  --report wenet/kenlm_lm/reports/lexicon_report.json \
  --tokenization bpe \
  --sentencepiece-model /path/to/the/am_training_tokenizer.model \
  --bpe-max-spellings 3 \
  --bpe-add-character-fallback \
  --ignore-case \
  --allow-rejected
```

In BPE mode the tool can emit up to `--bpe-max-spellings` unique spellings for
each LM word, in this order:

1. The canonical spelling produced by the SentencePiece model used to train
   the AM.
2. A globally shortest spelling found in `tokens.txt`. This uses dynamic
   programming; it does not use the old greedy longest-prefix segmentation.
3. A character fallback when `--bpe-add-character-fallback` is enabled and all
   required character tokens exist.

For example, the GigaSpeech tokenizer spells `CIRCULATION` as
`▁C IR C ULATION`. The old greedy method selected `▁C IR CU LA TION`, which is
valid token-by-token but may not match the AM path well. Keeping bounded,
deduplicated alternatives lets the decoder accept the canonical AM path while
avoiding an unbounded lexicon expansion.

The tool validates the SentencePiece vocabulary against `tokens.txt` and
rejects an obviously mismatched tokenizer (coverage below 99%). The JSON report
records tokenizer coverage, total lexicon entries, spelling-count distribution,
and rejected words. `--allow-rejected` is useful when the LM contains words the
AM token set cannot spell, such as digits or markup; those words are skipped and
listed in the report.

Use the exact tokenizer that produced the AM labels. A tokenizer with the same
vocabulary size is not necessarily compatible. The Python environment running
the command must have the `sentencepiece` package installed.

Decoder scoring can be tuned without editing the script:

```bash
LM_WEIGHT=0.5 WORD_SCORE=0.0 BEAM_SIZE=30 BEAM_SIZE_TOKEN=15 \
  wenet/kenlm_lm/tools/run_003_acceptance_eval.sh
```

To make the public SDK package use the same tuned settings, rebuild the package
with the same environment variables:

```bash
LM_WEIGHT=0.5 WORD_SCORE=0.0 BEAM_SIZE=30 BEAM_SIZE_TOKEN=15 \
  wenet/SDK/0.0.3/scripts/prepare_flashlight_runtime_package.sh
```

For standard sherpa/icefall Zipformer CTC models, add `FEATURE_TYPE=kaldi`.
Use `FEATURE_TYPE=whisper` only for packages that were validated with the older
Whisper-style frontend.

Score public SDK batch output against the same sampled references:

```bash
python3 wenet/kenlm_lm/tools/score_sdk_results.py \
  --result test/0.0.3/sdk_batch_flashlight.txt
```
