# Local Flashlight-Text patches

`flashlight/lib/text/decoder/Trie.{h,cpp}` keeps the historical six-label
value only as a vector reserve hint.  The original hard cap silently discarded
valid labels on a shared acoustic spelling, which breaks homophones and
runtime contact ambiguity.  A `RUNTIME_CONTACT_MAX` smearing mode takes the
maximum across labels at a completed-prefix node instead of log-adding them,
so a shared contact prefix does not become stronger merely because it has more
labels; the legacy `MAX` behavior remains intact for no-context decoding.  No
other trie behavior is changed.

`flashlight/lib/text/decoder/lm/KenLM.{h,cpp}` adds `HasWord()`, used only to
validate that a declared runtime contact class word is actually present in the
KenLM vocabulary rather than being scored as `<unk>`.
