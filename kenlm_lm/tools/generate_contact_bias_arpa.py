#!/usr/bin/env python3
"""Generate a small KenLM-compatible ARPA for runtime contact bias.

Rules are tab-separated ``PATTERN<TAB>BONUS`` entries. Every non-comment
pattern must contain exactly one trailing ``<CONTACT>``. The user-facing
bonus is positive, but its terminal ARPA n-gram is written as its negative so
KenLM accepts it. SDK 0.0.6 recognizes verified terminal matches and turns
that encoded value back into an additive positive contact bonus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple


FORMAT = "pattern_bias_v1"
CONTACT = "<CONTACT>"
BEGIN = "<s>"
END = "</s>"
UNKNOWN = "<unk>"


def fail(message: str) -> None:
    raise ValueError(message)


def normalize_pattern(value: str, line_no: int) -> Tuple[str, ...]:
    tokens = value.upper().split()
    if not tokens:
        fail(f"line {line_no}: pattern is empty")
    if tokens.count(CONTACT) != 1:
        fail(f"line {line_no}: pattern must contain exactly one {CONTACT}")
    if tokens[-1] != CONTACT:
        fail(f"line {line_no}: {CONTACT} must be the final token")
    for token in tokens:
        if token in {BEGIN, END, UNKNOWN}:
            fail(f"line {line_no}: reserved KenLM token is not allowed: {token}")
    return tuple(tokens)


def load_rules(path: Path) -> Dict[Tuple[str, ...], float]:
    rules: Dict[Tuple[str, ...], float] = {}
    with path.open("r", encoding="utf-8") as source:
        for line_no, raw in enumerate(source, 1):
            line = raw.rstrip("\n")
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) != 2:
                fail(f"line {line_no}: expected exactly PATTERN<TAB>BONUS")
            pattern = normalize_pattern(fields[0], line_no)
            try:
                bonus = float(fields[1].strip())
            except ValueError as exc:
                raise ValueError(f"line {line_no}: bonus is not numeric") from exc
            if not math.isfinite(bonus) or bonus <= 0.0:
                fail(f"line {line_no}: bonus must be finite and > 0")
            previous = rules.get(pattern)
            if previous is not None and previous != bonus:
                fail(
                    f"line {line_no}: duplicate pattern has conflicting "
                    f"bonus {previous} vs {bonus}"
                )
            rules[pattern] = bonus
    if not rules:
        fail("rules file has no usable rules")
    return rules


def all_subgrams(tokens: Tuple[str, ...]) -> Iterable[Tuple[str, ...]]:
    for size in range(1, len(tokens) + 1):
        for begin in range(0, len(tokens) - size + 1):
            yield tokens[begin : begin + size]


def arpa_ngrams(
    rules: Dict[Tuple[str, ...], float]
) -> Dict[int, Dict[Tuple[str, ...], float]]:
    ngrams: Dict[int, Dict[Tuple[str, ...], float]] = defaultdict(dict)
    for token in (UNKNOWN, BEGIN, END, CONTACT):
        ngrams[1][(token,)] = 0.0
    for pattern, bonus in rules.items():
        # A bare <CONTACT> means only a complete one-name utterance. The
        # explicit begin marker lets the SDK distinguish it from fallback.
        encoded = (BEGIN, CONTACT) if pattern == (CONTACT,) else pattern
        for gram in all_subgrams(encoded):
            ngrams[len(gram)].setdefault(gram, 0.0)
        ngrams[len(encoded)][encoded] = -bonus
    return ngrams


def format_score(value: float) -> str:
    # All values are <= 0 by construction. Avoid -0.0 because zero denotes a
    # scaffold/backoff transition rather than a matched terminal rule.
    if value == 0.0:
        return "0"
    return format(value, ".12g")


def write_arpa(
    path: Path, ngrams: Dict[int, Dict[Tuple[str, ...], float]]
) -> None:
    max_order = max(ngrams)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        output.write("\\data\\\n")
        for order in range(1, max_order + 1):
            output.write(f"ngram {order}={len(ngrams.get(order, {}))}\n")
        for order in range(1, max_order + 1):
            output.write(f"\n\\{order}-grams:\n")
            for gram, score in sorted(ngrams.get(order, {}).items()):
                fields = [format_score(score), " ".join(gram)]
                if order < max_order:
                    fields.append("0")
                output.write("\t".join(fields) + "\n")
        output.write("\n\\end\\\n")


def canonical_rules(rules: Dict[Tuple[str, ...], float]) -> str:
    return "".join(
        f"{' '.join(pattern)}\t{format_score(bonus)}\n"
        for pattern, bonus in sorted(rules.items())
    )


def write_metadata(
    path: Path, rules: Dict[Tuple[str, ...], float], max_order: int
) -> None:
    canonical = canonical_rules(rules).encode("utf-8")
    payload = {
        "format": FORMAT,
        "max_bonus": max(rules.values()),
        "max_order": max_order,
        "rule_count": len(rules),
        "rules_sha256": hashlib.sha256(canonical).hexdigest(),
        "standalone_rule": (CONTACT,) in rules,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules", required=True, type=Path,
                        help="TSV file containing PATTERN<TAB>BONUS rules")
    parser.add_argument("--arpa", required=True, type=Path,
                        help="output handmade ARPA path")
    parser.add_argument("--metadata", required=True, type=Path,
                        help="output pattern-bias metadata JSON path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rules = load_rules(args.rules)
        ngrams = arpa_ngrams(rules)
        write_arpa(args.arpa, ngrams)
        write_metadata(args.metadata, rules, max(ngrams))
    except (OSError, ValueError) as exc:
        print(f"generate_contact_bias_arpa.py: {exc}", file=sys.stderr)
        return 1
    print(
        f"generated {args.arpa} and {args.metadata}: "
        f"{len(rules)} rules, max_bonus={max(rules.values()):g}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
