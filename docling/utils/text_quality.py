# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Detect a PDF text layer that does not match the text drawn on the page.

The checks look at a page's whole text layer and at each of its lines, and only
at patterns broken text produces, such as unmapped font encodings.
"""

import math
import unicodedata
from itertools import groupby

_MIN_CHARS = 20  # Less text than this is left alone.
_MIN_LATIN_LETTERS = 50  # Too few Latin letters to judge their mix.
_CONTROL_SHARE = 0.03  # Control or U+FFFD characters.
_PRIVATE_USE_SHARE = 0.5  # Private-use characters; icon fonts stay far below.
_NON_ASCII_LATIN_SHARE = 0.5  # Latin letters outside ASCII.
_ASCII_LETTER_SHARE = 1 / 3  # Mojibake leaves few ASCII letters.

# One line is one text cell.
_PRIVATE_USE_RUN = 4  # Different private-use characters in a row.
_MIN_LINE_CHARS = 5  # Less text than this in a line or token is not judged.
_TOKEN_CONTROLS = 2  # Control characters in one token.
_TOKEN_CONTROL_SHARE = 0.05

# U+FFFD on the page.
_SHORT_PAGE_CHARS = 80  # On a page this short, a run of 2 is enough.
_SHORT_PAGE_RUN = 2
_MANY_REPLACEMENTS = 12
_MANY_REPLACEMENTS_SHARE = 0.05
_REPLACEMENT_LINES = 3
_LONG_REPLACEMENT_RUN = 8
_SOME_REPLACEMENTS_SHARE = 0.025

# Letters shifted by a broken ToUnicode map, as in a substitution cipher.
_MIN_CIPHER_LETTERS = 200  # Too few letters to judge their frequencies.
_MAX_VOWEL_SHARE = 0.30  # Latin-script text stays above this.
_MAX_ENGLISH_COSINE = 0.60  # Letters not where English puts them...
_MIN_SHAPE_COSINE = 0.90  # ...but with the frequency profile of a language.
# English letter frequencies, percent, a to z (Lewand, "Cryptological Mathematics").
_ENGLISH_LETTER_FREQ = (
    8.2, 1.5, 2.8, 4.3, 12.7, 2.2, 2.0, 6.1, 7.0, 0.15, 0.8, 4.0, 2.4,
    6.7, 7.5, 1.9, 0.1, 6.0, 6.3, 9.1, 2.8, 1.0, 2.4, 0.15, 2.0, 0.07,
)  # fmt: skip

# Not "Cn" (unassigned): that depends on the Unicode version of the Python running.
_CONTROL = frozenset({"Cc", "Cs"})

# Fonts with no Unicode map, as docling-parse writes them.
_MIN_GLYPH_CODES = 20  # Glyph codes written as decimal numbers, one per glyph.
_GLYPH_CODE_SHARE = 0.6  # ...of all tokens.
_MIN_GLYPH_CODE = 20  # Median code; note markers stay below it.
_MIN_GLYPH_CODE_KINDS = 10  # One code per letter shape; a table of years has few.
_MIN_GLYPH_NAME_WORDS = 8  # Words spelled in Arabic glyph names (`qafisolated`).
# Fonts that spread one word over several subsets come out one character per cell.
_MIN_FRAGMENTS = 40
_FRAGMENT_SHARE = 0.6
_MIN_FRAGMENT_KINDS = 10  # A bracket or bullet font repeats a few characters.
_LEADERS = frozenset(".-_*=~")

# Arabic glyph names in fonts (Adobe Glyph List style, glued per word by docling-parse):
# a letter, optionally `with` another letter (ligatures), then positional suffixes.
_ARABIC_GLYPHS = (
    "alef beh teh theh jeem hah khah dal thal reh zain seen sheen sad dad tah zah ain "
    "ghain feh qaf kaf lam meem noon heh waw yeh hamza tehmarbuta alefmaksura tatweel "
    "allah alefwasla peh tcheh jeh veh keheh gaf farsiyeh fathatan dammatan kasratan "
    "fatha damma kasra shadda sukun superscriptalef arabiccomma arabicsemicolon "
    "arabicquestionmark alefwithhamzaabove alefwithhamzabelow alefwithmaddaabove "
    "wawwithhamzaabove yehwithhamzaabove alefhamzaabove alefhamzabelow alefmaddaabove "
    + " ".join(
        f"arabicindicdigit{d}"
        for d in "zero one two three four five six seven eight nine".split()
    )
).split()
_GLYPH_SUFFIXES = ("isolated", "initial", "medial", "final", "low")


def _share(text: str, categories: frozenset[str], *, replacement: bool) -> float:
    """Share of non-space characters in the given Unicode categories."""
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return 0.0
    hits = sum(
        unicodedata.category(ch) in categories or (replacement and ch == "\ufffd")
        for ch in chars
    )
    return hits / len(chars)


def _is_private_use(ch: str) -> bool:
    return unicodedata.category(ch) == "Co"


def _non_ascii_latin_share(text: str) -> float:
    """Share of Latin letters outside ASCII, on text where Latin letters dominate."""
    letters = [ch for ch in text if ch.isalpha()]
    latin = [
        ch
        for ch in letters
        if "LATIN" in unicodedata.name(ch, "") and not "\uff00" <= ch <= "\uffef"
    ]
    if len(latin) < _MIN_LATIN_LETTERS or len(latin) * 2 < len(letters):
        return 0.0
    return sum(ord(ch) > 127 for ch in latin) / len(latin)


def _ascii_letter_share(text: str) -> float:
    """Share of non-space characters that are ASCII letters."""
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return 0.0
    return sum(ch.isascii() and ch.isalpha() for ch in chars) / len(chars)


def _is_broken_line(line: str) -> bool:
    """Whether one line holds a run or a token that only a broken font produces."""
    # Icons repeat a few private-use code points; the letters of words vary.
    for is_private_use, run in groupby(line, key=_is_private_use):
        if is_private_use and len(set(run)) >= _PRIVATE_USE_RUN:
            return True
    for token in line.split():
        if len(token) < _MIN_LINE_CHARS:
            continue
        controls = sum(unicodedata.category(ch) in _CONTROL for ch in token)
        if controls >= _TOKEN_CONTROLS and controls >= _TOKEN_CONTROL_SHARE * len(
            token
        ):
            return True
    return False


def _has_replacement_evidence(text: str, lines: list[str]) -> bool:
    """Whether U+FFFD is dense or clustered enough to mark the page broken."""
    count = text.count("\ufffd")
    if not count:
        return False
    chars = sum(not ch.isspace() for ch in text)
    longest = max((len(list(run)) for ch, run in groupby(text) if ch == "\ufffd"))
    if chars <= _SHORT_PAGE_CHARS and longest >= _SHORT_PAGE_RUN:
        return True
    share = count / chars
    lines_hit = sum("\ufffd" in line for line in lines)
    return (
        (count >= _MANY_REPLACEMENTS and share >= _MANY_REPLACEMENTS_SHARE)
        or (lines_hit >= _REPLACEMENT_LINES and share >= _SOME_REPLACEMENTS_SHARE)
        or (longest >= _LONG_REPLACEMENT_RUN and share >= _SOME_REPLACEMENTS_SHARE)
    )


def _cosine(a: list[float], b: list[float]) -> float:
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b)) / norm if norm else 1.0


def _is_shifted_latin(text: str) -> bool:
    """Whether ASCII letters look like language with its letters swapped.

    A broken ToUnicode map often shifts every letter by a constant. That keeps
    the shape of the letter-frequency profile but moves each letter. Only tokens
    with a lowercase letter count, so acronyms and part numbers are skipped.
    """
    counts = [0.0] * 26
    other_letters = 0
    for token in text.split():
        if not any("a" <= ch <= "z" for ch in token):
            continue
        for ch in token:
            if "a" <= ch.lower() <= "z":
                counts[ord(ch.lower()) - ord("a")] += 1
            elif ch.isalpha():
                other_letters += 1
    letters = sum(counts)
    if letters < _MIN_CIPHER_LETTERS or other_letters > letters:
        return False
    vowels = sum(counts[ord(v) - ord("a")] for v in "aeiou")
    if vowels / letters > _MAX_VOWEL_SHARE:
        return False
    english = list(_ENGLISH_LETTER_FREQ)
    return (
        _cosine(counts, english) < _MAX_ENGLISH_COSINE
        and _cosine(sorted(counts, reverse=True), sorted(english, reverse=True))
        >= _MIN_SHAPE_COSINE
    )


def _has_glyph_codes(tokens: list[str]) -> bool:
    """Whether most tokens are glyph codes: integers that repeat, as every alef has the
    same code, but take many values, one per letter shape, above the range of note
    markers. Page numbers do not repeat; a table of years takes few values."""
    numbers = [int(t) for t in tokens if t.isascii() and t.isdigit() and len(t) <= 5]
    if len(numbers) < _MIN_GLYPH_CODES or len(numbers) <= _GLYPH_CODE_SHARE * len(
        tokens
    ):
        return False
    letters = sum(ch.isalpha() for t in tokens for ch in t)
    kinds = len(set(numbers))
    return (
        _MIN_GLYPH_CODE_KINDS <= kinds <= len(numbers) / 2
        and sorted(numbers)[len(numbers) // 2] >= _MIN_GLYPH_CODE
        and letters < len(numbers)
    )


def _glyph_count(token: str) -> int:
    """How many Arabic glyph names spell the token, or 0 if it is not spelled so."""
    best = [0] + [-1] * len(token)  # best[i]: fewest glyphs that spell token[:i]
    for i in range(len(token)):
        if best[i] < 0:
            continue
        ends = [i + len(g) for g in _ARABIC_GLYPHS if token.startswith(g, i)]
        while ends:
            end = ends.pop()
            if best[end] < 0 or best[end] > best[i] + 1:
                best[end] = best[i] + 1
            for suffix in _GLYPH_SUFFIXES:
                if token.startswith(suffix, end):
                    ends.append(end + len(suffix))
            if token.startswith("with", end):
                ends += [
                    end + 4 + len(g)
                    for g in _ARABIC_GLYPHS
                    if token.startswith(g, end + 4)
                ]
            # Alternate glyphs: `isolatedd`, `salla2`.
            if end < len(token) and token[end] in "abcd0123456789":
                if best[end + 1] < 0 or best[end + 1] > best[i] + 1:
                    best[end + 1] = best[i] + 1
    return max(best[-1], 0)


def _has_glyph_names(tokens: list[str]) -> bool:
    """Whether the text spells glyphs by their names (`yehfinalfehinitial`). One short
    name could be an English word (`seen`); two glyphs or a suffix make it a name."""
    named = 0
    for token in tokens:
        if (
            token.isascii()
            and token.islower()
            and len(token) > 8
            and _glyph_count(token)
        ):
            named += 1
    return named >= _MIN_GLYPH_NAME_WORDS


def _has_letter_fragments(tokens: list[str]) -> bool:
    """Whether the text is mostly single ASCII characters of many kinds, as a legacy font
    split across subsets comes out when cells break at every font change."""
    singles = [
        t
        for t in tokens
        if len(t) == 1 and t.isascii() and not t.isdigit() and t not in _LEADERS
    ]
    words = sum(len(t) > 1 and any(ch.isalpha() for ch in t) for t in tokens)
    return (
        len(singles) >= _MIN_FRAGMENTS
        and len(singles) > _FRAGMENT_SHARE * len(tokens)
        and len(singles) > 4 * words
        and len(set(singles)) >= _MIN_FRAGMENT_KINDS
    )


def is_broken_text_layer(text: str) -> bool:
    """Whether a page's PDF text does not match the text drawn on the page.

    Args:
        text: All text of one page's PDF text layer, cells joined by newlines.

    Returns:
        ``True`` only for text a broken layer produces. When unsure it returns
        ``False``, so pages keep their PDF text.
    """
    stripped = text.strip()
    if len(stripped) < _MIN_CHARS:
        return False
    lines = stripped.splitlines()

    # A font whose glyphs were mapped to control or private-use code points.
    if _share(stripped, _CONTROL, replacement=True) >= _CONTROL_SHARE:
        return True
    if _share(stripped, frozenset({"Co"}), replacement=False) >= _PRIVATE_USE_SHARE:
        return True
    if _has_replacement_evidence(stripped, lines):
        return True
    # The same, in one cell of an otherwise good page.
    if any(_is_broken_line(line) for line in lines):
        return True

    # A non-Latin font whose glyphs were mapped to accented Latin letters.
    if (
        _non_ascii_latin_share(stripped) >= _NON_ASCII_LATIN_SHARE
        and _ascii_letter_share(stripped) < _ASCII_LETTER_SHARE
    ):
        return True

    # A font with no Unicode map, written as glyph codes, glyph names or one character
    # per cell.
    tokens = stripped.split()
    if (
        _has_glyph_codes(tokens)
        or _has_glyph_names(tokens)
        or _has_letter_fragments(tokens)
    ):
        return True

    # A Latin font whose glyphs were mapped to the wrong ASCII letters.
    return _is_shifted_latin(stripped)
