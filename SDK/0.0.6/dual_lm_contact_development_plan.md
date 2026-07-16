# Development Plan: Dual-LM Runtime Contact Decoding

## Status

This is the active SDK 0.0.6 contact-recognition design. It supersedes
`runtime_contact_bias_development_plan.md`.

## Objective

Recognize a runner-provided contact list without manual contact rewards. The
main LM scores ordinary text and a separately trained contact LM scores a
normalized contact-domain hypothesis. The acoustic model runs once; the two
decoders consume the same CTC emissions.

```text
main score    = AM + main_lm_weight × LM_main(original words)
contact score = AM + contact_lm_weight × LM_contact(normalized words)
```

Each complete runtime name is one dynamic lexicon label. The AM consumes its
real token sequence, but the contact LM advances once with `<CONTACT>`. Thus
all names receive the same LM score in the same history; only AM evidence and
normal decoder bookkeeping distinguish them.

## Package and LM preparation

1. Normalize every complete annotated contact span to one whitespace-delimited
   `<CONTACT>` token before contact-LM training. A multiword name must become
   one placeholder, never one placeholder per name word.
2. Train `contact_lm.bin` on the normalized contact-domain text. Train the
   ordinary `lm.bin` as the main LM.
3. Build one shared `words.txt` from the union of main-LM and contact-LM text,
   then build one shared static `lexicon.txt`. Include `<CONTACT>` in
   `words.txt`, but explicitly exclude it from the AM token table and static
   lexicon.
4. A contact-capable manifest contains `contact_lm`, `contact_lm_weight`, and
   `contact_class_word` in addition to the existing main `lm` and `lm_weight`.
   A package without `contact_lm` remains main-only and rejects non-empty
   contact contexts.

The package workflow accepts `MAIN_LM_BIN`, optional `CONTACT_LM_BIN`,
`WORDS_FILE`, `LEXICON_FILE`, and `CONTACT_LM_WEIGHT`. It copies the second LM
and records it in both the manifest and checksums.

## Runtime behavior

1. Compile the user contact list once into an immutable trie shared by streams.
   The trie stores complete AM token spellings and unique dynamic contact IDs.
2. Construct a base main decoder and a contact decoder. The contact decoder
   uses the base lexicon plus the contact trie and its own LM weight.
3. On ordinary words, the contact decoder advances the contact LM. While AM
   tokens are inside a contact spelling, the contact-LM state stays at the
   preceding ordinary-word history. On a complete contact terminal, the
   wrapper maps its dynamic ID to `<CONTACT>` and advances the contact LM once.
4. Use MAX-smearing and contact-LM lookahead for incomplete names. The terminal
   transition subtracts that temporary lookahead and adds the exact
   history-dependent class score, so lookahead affects pruning only.
5. At each partial/final update, merge the main beam with contact-beam paths
   that contain at least one completed runtime contact. Contact paths without a
   completed contact cannot win cross-decoder selection. Sort by weighted total
   score, retain contact-aware semantic distinctions, and apply `nbest` only
   after merging.
6. Report each entity score as
   `contact_lm_weight × log P_contact(<CONTACT> | normalized history)`.
   Replay decoded normalized words through the contact LM when mapping results
   so this score remains local to each entity.

## API and compatibility

- `ContactListOptions` contains contact-list limits only. Trigger and manual
  score configuration are removed.
- The C score/trigger functions and CLI reward flags are removed. The C ABI is
  version 3.
- Contact and spoken-form APIs remain runtime-only. Contact data is never
  written to a model package or debug log.
- Empty contexts use only the original main decoder and must preserve baseline
  output and scores exactly.

## Verification

- Unit-test dynamic-ID-to-`<CONTACT>` mapping, multiword names, overlapping
  names, failed prefixes, and equal class-LM scores for different contacts.
- Test independent LM weights, contact-required beam eligibility, multiple
  contacts, contact entity score replay, package validation, and main-only
  fallback.
- Benchmark 0, 1k, 10k, and 50k contact forms. Record context compile time,
  trie memory, decoder-only RTF, end-to-end RTF, and peak beam size. Per-frame
  work must traverse active trie states only; it must never scan the full list.
