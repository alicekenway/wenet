from pathlib import Path

from pynini import closure, cross, string_file, union
from pynini.lib.pynutil import delete, insert

from tn.processor import Processor


class Radio(Processor):
    """Canonicalize spoken broadcast bands and frequencies."""

    def __init__(self, cardinal, decimal):
        super().__init__(name="radio", ordertype="itn")
        self.cardinal = cardinal
        self.decimal = decimal
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        data = Path(__file__).resolve().parents[1] / "data" / "radio" / "band.tsv"
        band = string_file(str(data))
        ds = delete(" ")
        digit_words = union(
            cross("zero", "0"), cross("oh", "0"), cross("o", "0"),
            cross("one", "1"), cross("two", "2"), cross("three", "3"),
            cross("four", "4"), cross("five", "5"), cross("six", "6"),
            cross("seven", "7"), cross("eight", "8"), cross("nine", "9"),
        )
        fraction = digit_words + closure(ds + digit_words)
        spoken_digits = digit_words + closure(ds + digit_words)
        integer = self.cardinal.graph | spoken_digits
        decimal = integer + cross(" point ", ".") + fraction
        pair = self.cardinal.graph_two_digit
        year_style = pair + ds + (pair | insert("0") + digit_words)
        hundreds_style = digit_words + ds + pair
        number = (
            decimal | year_style | hundreds_style | integer
            | cross("a hundred", "100")
        )
        unit = closure(
            ds + delete(union("megahertz", "kilohertz", "hertz", "mhz", "khz")),
            0, 1,
        )
        value = band + ds + number + unit
        value |= number + unit + ds + band

        # Some control utterances omit the broadcast band or put it earlier:
        # "tune to nine five point five" and
        # "tune to the f m station nine five point five".
        # Include the intent phrase in this token so digit-by-digit readings
        # are not interpreted as unrelated cardinals by the generic grammar.
        value |= union("tune to", "play") + " " + number + unit
        value |= "tune to the " + band + " station " + number + unit
        self.tagger = self.add_tokens(insert('value: "') + value + insert('"'))

    def build_verbalizer(self):
        value = delete('value: "') + self.NOT_QUOTE.plus + delete('"')
        self.verbalizer = self.delete_tokens(value)
