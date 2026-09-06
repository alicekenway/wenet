from pynini import cross
from pynini.lib.pynutil import delete, insert

from tn.processor import Processor


class AmpersandSequence(Processor):
    """Restore supported spoken letter sequences containing an ampersand."""

    def __init__(self):
        super().__init__(name="ampersand_sequence", ordertype="itn")
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        value = cross("r n b", "R&B")
        self.tagger = self.add_tokens(insert('value: "') + value + insert('"'))

    def build_verbalizer(self):
        value = delete('value: "') + self.NOT_QUOTE.plus + delete('"')
        self.verbalizer = self.delete_tokens(value)
