from pynini import closure, cross, union
from pynini.lib.pynutil import add_weight, delete, insert

from tn.processor import Processor


def preferred_compound_two_digit():
    """Map TWENTY ONE..NINETY NINE as one preferred numeric token."""
    tens = union(
        cross("twenty", "2"), cross("thirty", "3"),
        cross("forty", "4"), cross("fifty", "5"),
        cross("sixty", "6"), cross("seventy", "7"),
        cross("eighty", "8"), cross("ninety", "9"),
    )
    ones = union(
        cross("one", "1"), cross("two", "2"), cross("three", "3"),
        cross("four", "4"), cross("five", "5"), cross("six", "6"),
        cross("seven", "7"), cross("eight", "8"), cross("nine", "9"),
    )
    # A negative preference is safe here because this graph always consumes
    # exactly two non-empty words. It prevents the global classifier from
    # choosing separate TENS and ONES tokens.
    return add_weight(tens + delete(" ") + ones, -2.0)


class ControlToken(Processor):
    """Normalize compact control-domain tokens and unit fractions."""

    def __init__(self):
        super().__init__(name="control_token", ordertype="itn")
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        graph = union(
            cross("a c", "A/C"),
            cross("one half", "1/2"),
            cross("a half", "1/2"),
            cross("one third", "1/3"),
            cross("a third", "1/3"),
            cross("one quarter", "1/4"),
            cross("a quarter", "1/4"),
            cross("to half", "to 1/2"),
            cross("to quarter", "to 1/4"),
            cross("three sixty camera", "360 camera"),
        )
        self.tagger = self.add_tokens(
            insert('value: "') + graph + insert('"')
        )

    def build_verbalizer(self):
        value = delete('value: "') + self.NOT_QUOTE.plus + delete('"')
        self.verbalizer = self.delete_tokens(value)


class CompactPercent(Processor):
    """Render spoken percentages without a space before the percent sign."""

    def __init__(self, cardinal):
        super().__init__(name="compact_percent", ordertype="itn")
        self.cardinal = cardinal
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        number = (
            preferred_compound_two_digit()
            | self.cardinal.graph
            | cross("a hundred", "100")
        )
        graph = number + delete(" ") + cross("percent", "%")
        self.tagger = self.add_tokens(
            insert('value: "') + graph + insert('"')
        )

    def build_verbalizer(self):
        value = delete('value: "') + self.NOT_QUOTE.plus + delete('"')
        self.verbalizer = self.delete_tokens(value)


class ControlNumber(Processor):
    """Normalize numbers only in unambiguous control-setting contexts."""

    def __init__(self, cardinal):
        super().__init__(name="control_number", ordertype="itn")
        self.cardinal = cardinal
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        cardinal_number = (
            preferred_compound_two_digit()
            | self.cardinal.graph
            | cross("a hundred", "100")
        )
        ds = delete(" ")
        fraction_digit = union(
            cross("zero", "0"), cross("oh", "0"), cross("o", "0"),
            cross("one", "1"), cross("two", "2"), cross("three", "3"),
            cross("four", "4"), cross("five", "5"), cross("six", "6"),
            cross("seven", "7"), cross("eight", "8"), cross("nine", "9"),
        )
        decimal_number = (
            cardinal_number + cross(" point ", ".")
            + fraction_digit + closure(ds + fraction_digit)
        )
        number = decimal_number | cardinal_number

        # Preserve product-facing unit words rather than abbreviating SECONDS
        # to "s" through the generic measurement grammar.
        unit = union(
            "level", "levels", "step", "steps", "notch", "notches",
            "second", "seconds",
        )
        number_with_unit = number + " " + unit

        # These immediate contexts are numeric throughout the control corpus.
        by_number = "by " + number
        by_number_with_unit = by_number + " " + unit
        arithmetic_number = union("add", "subtract") + " " + number
        to_context = union(
            "brightness", "volume", "sound", "level", "intensity",
            "strength", "seat",
        )
        context_to_number = to_context + " to " + number
        context_to_number_with_unit = context_to_number + " " + unit
        context_to_percent = (
            context_to_number + delete(" ") + cross("percent", "%")
        )
        direct_context = union(
            "brightness", "volume", "level", "step", "phone",
        )
        context_number = direct_context + " " + number
        context_number_with_unit = context_number + " " + unit

        graph = union(
            number_with_unit,
            by_number_with_unit,
            by_number,
            arithmetic_number,
            context_to_number_with_unit,
            context_to_percent,
            context_to_number,
            context_number_with_unit,
            context_number,
        )
        self.tagger = self.add_tokens(
            insert('value: "') + graph + insert('"')
        )

    def build_verbalizer(self):
        value = delete('value: "') + self.NOT_QUOTE.plus + delete('"')
        self.verbalizer = self.delete_tokens(value)


class ControlTelephone(Processor):
    """Normalize digit-spoken phone requests with product formatting."""

    def __init__(self):
        super().__init__(name="control_telephone", ordertype="itn")
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        ds = delete(" ")
        digit = union(
            cross("zero", "0"), cross("oh", "0"), cross("o", "0"),
            cross("one", "1"), cross("two", "2"), cross("three", "3"),
            cross("four", "4"), cross("five", "5"), cross("six", "6"),
            cross("seven", "7"), cross("eight", "8"), cross("nine", "9"),
        )

        # Accept 3–15 digits compactly. The lower-weight exact-ten path wins
        # for ten digits and formats them as XXX-XXX-XXXX.
        compact = digit + closure(ds + digit, 2, 14)
        dashed_ten = (
            digit + ds + digit + ds + digit + insert("-")
            + ds + digit + ds + digit + ds + digit + insert("-")
            + ds + digit + ds + digit + ds + digit + ds + digit
        )
        number = dashed_ten | add_weight(compact, 0.01)
        number = cross("plus", "+") + ds + number | number

        # Keeping the intent word inside this token prevents ordinary quantity
        # sequences from being interpreted as telephone numbers.
        context = union("call", "call to", "dial", "phone", "phone number")
        graph = context + " " + number
        self.tagger = self.add_tokens(
            insert('value: "') + graph + insert('"')
        )

    def build_verbalizer(self):
        value = delete('value: "') + self.NOT_QUOTE.plus + delete('"')
        self.verbalizer = self.delete_tokens(value)
