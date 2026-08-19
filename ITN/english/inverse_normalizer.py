from pathlib import Path

from pynini import closure
from pynini.lib.pynutil import add_weight, delete

from itn.english.inverse_normalizer import InverseNormalizer as UpstreamInverseNormalizer
from itn.english.rules.cardinal import Cardinal
from itn.english.rules.char import Char
from itn.english.rules.date import Date
from itn.english.rules.decimal import Decimal
from itn.english.rules.electronic import Electronic
from itn.english.rules.measure import Measure
from itn.english.rules.money import Money
from itn.english.rules.ordinal import Ordinal
from itn.english.rules.punctuation import Punctuation
from itn.english.rules.telephone import Telephone
from itn.english.rules.time import Time
from itn.english.rules.whitelist import Whitelist
from itn.english.rules.word import Word

from .rules.context_number import ContextNumber
from .rules.identifier import Identifier
from .rules.radio import Radio


class InverseNormalizer(UpstreamInverseNormalizer):
    """Upstream English ITN plus conservative product rules."""

    def __init__(self, cache_dir=None, overwrite_cache=False):
        if cache_dir is None:
            cache_dir = Path(__file__).resolve().parent / "export"
        super().__init__(cache_dir=str(cache_dir), overwrite_cache=overwrite_cache)

    def build_tagger_and_verbalizer(self):
        cardinal = Cardinal()
        ordinal = Ordinal(cardinal=cardinal)
        decimal = Decimal(cardinal=cardinal)
        date = Date(cardinal=cardinal, ordinal=ordinal)
        time = Time(cardinal=cardinal)
        measure = Measure(cardinal=cardinal, decimal=decimal)
        money = Money(cardinal=cardinal, decimal=decimal)
        telephone = Telephone(cardinal=cardinal)
        electronic = Electronic()
        whitelist = Whitelist()
        word = Word()
        char = Char()
        punctuation = Punctuation()
        radio = Radio(cardinal, decimal)
        identifier = Identifier(cardinal)
        context_number = ContextNumber(cardinal)

        classify = (
            add_weight(radio.tagger, 0.1)
            | add_weight(context_number.tagger, 0.2)
            | add_weight(identifier.tagger, 0.3)
            | add_weight(date.tagger, 1.09)
            | add_weight(time.tagger, 1.1)
            | add_weight(measure.tagger, 1.1)
            | add_weight(money.tagger, 1.08)
            | add_weight(whitelist.tagger, 1.01)
            | add_weight(telephone.tagger, 1.1)
            | add_weight(electronic.tagger, 1.1)
            | add_weight(ordinal.tagger, 1.09)
            | add_weight(decimal.tagger, 1.1)
            | add_weight(cardinal.tagger, 1.1)
            | add_weight(word.tagger, 50)
            | add_weight(char.tagger, 100)
        ).optimize()
        punct = add_weight(punctuation.tagger, 1.1)
        token = closure(punct + delete(" ").ques) + classify + closure(delete(" ").ques + punct)
        graph = token + closure(self.DELETE_EXTRA_SPACE + token)
        self.tagger = delete(" ").star + graph + delete(" ").star

        verbalizer = (
            radio.verbalizer | context_number.verbalizer | identifier.verbalizer
            | cardinal.verbalizer | ordinal.verbalizer | decimal.verbalizer
            | date.verbalizer | time.verbalizer | measure.verbalizer
            | money.verbalizer | telephone.verbalizer | electronic.verbalizer
            | whitelist.verbalizer | word.verbalizer | char.verbalizer
            | punctuation.verbalizer
        ).optimize()
        self.verbalizer = (verbalizer + self.INSERT_SPACE).star @ self.build_rule(
            self.DELETE_EXTRA_SPACE
        ) @ self.build_rule(delete(" "), r="[EOS]")

    def normalize(self, input, nbest=1):
        # ASR vocabularies are often upper-case; semantic grammars are lower-case.
        output = super().normalize(input.lower(), nbest=nbest)
        if nbest != 1:
            return [self._restore_unchanged_case(input, item) for item in output]
        return self._restore_unchanged_case(input, output)

    @staticmethod
    def _restore_unchanged_case(source, output):
        source_lower = source.lower()
        cursor = 0
        chars = list(output)
        begin = 0
        while begin < len(chars):
            if not chars[begin].isalpha():
                begin += 1
                continue
            end = begin + 1
            while end < len(chars) and chars[end].isalpha():
                end += 1
            word = "".join(chars[begin:end])
            found = source_lower.find(word.lower(), cursor)
            if found >= 0 and len(source[found:found + len(word)]) == len(word):
                chars[begin:end] = source[found:found + len(word)]
                cursor = found + len(word)
            begin = end
        return "".join(chars)
