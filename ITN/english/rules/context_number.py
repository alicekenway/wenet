from pynini import closure, cross, union
from pynini.lib.pynutil import delete, insert

from tn.processor import Processor


class ContextNumber(Processor):
    """Convert short numbers only where product context makes intent clear."""

    def __init__(self, cardinal):
        super().__init__(name="context_number", ordertype="itn")
        self.cardinal = cardinal
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        ds = delete(" ")
        context = union("call", "dial", "phone", "page", "row", "level", "step", "room", "channel")
        digit = union(
            cross("zero", "0"), cross("oh", "0"), cross("o", "0"),
            cross("one", "1"), cross("two", "2"), cross("three", "3"),
            cross("four", "4"), cross("five", "5"), cross("six", "6"),
            cross("seven", "7"), cross("eight", "8"), cross("nine", "9"),
        )
        digit_sequence = digit + closure(ds + digit)
        number = digit_sequence | self.cardinal.graph
        # Keep the context word as ordinary text while normalizing its number.
        value = context + insert(" ") + ds + number
        self.tagger = self.add_tokens(insert('value: "') + value + insert('"'))

    def build_verbalizer(self):
        value = delete('value: "') + self.NOT_QUOTE.plus + delete('"')
        self.verbalizer = self.delete_tokens(value)
