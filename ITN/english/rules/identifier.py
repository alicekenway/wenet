from pynini import closure, cross, union
from pynini.lib.pynutil import delete, insert

from tn.processor import Processor


class Identifier(Processor):
    """Normalize compact alphanumeric product identifiers such as MH370."""

    def __init__(self, cardinal):
        super().__init__(name="identifier", ordertype="itn")
        self.cardinal = cardinal
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        ds = delete(" ")
        letters = union(*[cross(chr(c + 32), chr(c)) for c in range(ord("A"), ord("Z") + 1)])
        prefix = letters + closure(ds + letters, 1, 3)
        digit = union(
            cross("zero", "0"), cross("oh", "0"), cross("o", "0"),
            cross("one", "1"), cross("two", "2"), cross("three", "3"),
            cross("four", "4"), cross("five", "5"), cross("six", "6"),
            cross("seven", "7"), cross("eight", "8"), cross("nine", "9"),
        )
        pair = self.cardinal.graph_two_digit
        suffix = digit + ds + pair | pair | self.cardinal.graph
        graph = prefix + ds + suffix
        self.tagger = self.add_tokens(insert('value: "') + graph + insert('"'))

    def build_verbalizer(self):
        value = delete('value: "') + self.NOT_QUOTE.plus + delete('"')
        self.verbalizer = self.delete_tokens(value)
