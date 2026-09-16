#!/usr/bin/env python3
"""Compare automatic C++ word paths to the static Python builder, not hand paths."""
import argparse
import collections
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--tool', required=True)
parser.add_argument('--generator', required=True)
parser.add_argument('--package', type=Path, required=True)
parser.add_argument('--contacts', type=Path, required=True)
parser.add_argument('--report', type=Path, required=True)
args = parser.parse_args()
spec = importlib.util.spec_from_file_location('lexicon_generator', args.generator)
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)
tokens_path = args.package / 'tokens.txt'
model_path = args.package / 'sentencepiece.model'
token_to_id, _, vocab = gen.read_symbols(tokens_path)
token_set = {t for t in gen.valid_model_tokens(token_to_id, vocab)
             if not gen.is_special_token(t) and not gen.is_byte_token(t) and t != '▁'}
folded = gen.build_case_insensitive_index(token_set, token_to_id)
max_len = max(max(len(t), len(t.casefold())) for t in token_set)
processor = gen.load_sentencepiece_model(model_path)
words = {line.split()[0] for line in (args.package / 'words.txt').read_text().splitlines()
         if line.strip() and not line.startswith('<')}
contact_words = {word.upper() for name in args.contacts.read_text().splitlines()
                 for word in name.split()}
words.update(contact_words)
words.update(['Zara', 'davis', 'Straße', 'Élodie', 'éLODIE', 'Σ', 'ς', 'İ',
              'WANTS', 'DEFROST', '!!!', '𐐀', '😀', ''])
words = sorted(words)
run = subprocess.run([args.tool, str(tokens_path), str(model_path)],
                     input='\n'.join(words) + '\n', text=True, capture_output=True, check=True)
actual = {}
for line in run.stdout.splitlines():
    fields = line.split('\t')
    actual[fields[0]] = [tuple(map(int, path.split())) for path in fields[1:]]
assert set(actual) == set(words)
# Negative checks exercise real model loading and UTF-8 validation.
with tempfile.TemporaryDirectory(prefix='dcc-parity-') as temp:
    mismatch = Path(temp) / 'tokens.txt'
    mismatch.write_text('<blank> 0\n▁ 1\nA 2\n#0 3\n')
    bad = subprocess.run([args.tool, str(mismatch), str(model_path)],
                         input='ZARA\n', text=True, capture_output=True)
    assert bad.returncode != 0 and 'below 99%' in bad.stderr
bad_utf8 = subprocess.run([args.tool, str(tokens_path), str(model_path)],
                          input=b'\xff\n', capture_output=True)
assert bad_utf8.returncode != 0 and b'invalid UTF-8' in bad_utf8.stderr
histogram = collections.Counter()
for word in words:
    canonical, _ = gen.canonical_sentencepiece_spelling(processor, word, token_set, '▁', False)
    shortest, _, _ = gen.tokenize_bpe_shortest(word, token_set, token_to_id, max_len, '▁', folded)
    characters, _, _ = gen.tokenize_bpe_characters(word, token_set, '▁', False, folded)
    expected = []
    for path in (canonical, shortest, characters):
        if path is not None:
            ids = tuple(token_to_id[t] for t in path)
            if ids not in expected:
                expected.append(ids)
    assert actual[word] == expected, (word, actual[word], expected)
    histogram[len(expected)] += 1
for word in contact_words:
    assert actual[word], f'Unencodable contact word: {word}'
# Verify every word from the deployed static text lexicon has exactly the same
# path set. Special slot/unknown entries are not ordinary contact words.
static = collections.defaultdict(set)
for line in (args.package / 'lexicon.txt').read_text().splitlines():
    fields = line.split()
    if fields and not fields[0].startswith('<'):
        static[fields[0]].add(tuple(token_to_id[t] for t in fields[1:]))
for word, paths in static.items():
    assert set(actual[word]) == paths, ('static mismatch', word, actual[word], paths)
report = {'words_checked': len(words), 'static_words_checked': len(static),
          'contact_words_checked': len(contact_words), 'path_count_histogram': dict(histogram),
          'cpp_matches_python': True, 'cpp_matches_static_lexicon': True,
          'mismatched_tokenizer_rejected': True, 'invalid_utf8_rejected': True,
          'contact_word_paths': {w: actual[w] for w in sorted(contact_words)}}
args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({k: v for k, v in report.items() if k != 'contact_word_paths'}, indent=2))
