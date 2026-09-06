# Compact lexicon compiler

`compact_lexicon_tool` converts a validated SDK 0.0.14/0.0.15 Flashlight
source package into the portable `compact_trie_v1` binary used by SDK 0.0.16.
The existing Python lexicon generator remains authoritative for producing
`lexicon.txt`; this tool compiles that text and the package's LM-dependent
lookahead scores.

The executable is a Linux host build tool. It is built by SDK 0.0.16 when
`ASR_SDK_BUILD_TOOLS=ON`, but it is not installed, shipped in model packages,
or built for Android.

```bash
compact_lexicon_tool compile \
  --source-package /path/to/sdk-0.0.15-style-package \
  --output /path/to/lexicon.bin \
  --report /path/to/lexicon_report.json

compact_lexicon_tool verify \
  --source-package /path/to/sdk-0.0.15-style-package \
  --input /path/to/lexicon.bin

compact_lexicon_tool inspect --input /path/to/lexicon.bin
```

`compile` always performs the same exhaustive source-vs-binary verification
as `verify` before it succeeds. The compiler preserves every pronunciation,
homophone label, prefix word, and label insertion order.

## Portability and integrity

The file uses fixed-width little-endian integers and IEEE-754 float32 values.
It contains no native pointers, `size_t`, C++ structs, or ABI-dependent
padding. Every section starts at a 64-byte boundary. A single file is used by
Linux x86-64 and Android arm64-v8a, both of which are little-endian targets.

The payload checksum detects corruption. A dependency fingerprint binds the
file to `tokens.txt`, `words.txt`, `lm_search.json`, every configured LM,
pre/final mappings, and scoring settings. SDK 0.0.16 rejects a binary whose
fingerprint or dictionary metadata differs from the package.

## Runtime arrays

Nodes are numbered breadth-first. Edge `i` always targets node `i + 1`, so no
target array is stored. The binary contains:

- node-to-edge offsets and sorted 16-bit token IDs;
- node-to-label offsets and 32-bit word IDs;
- normal and runtime-contact lookahead scores for every node;
- word-to-pronunciation and pronunciation-to-token offsets;
- packed 16-bit pronunciation token IDs.

The SDK memory-maps these arrays read-only, with a one-buffer read fallback if
`mmap` is unavailable. Child lookup is linear for up to eight children and
binary search for larger nodes.
