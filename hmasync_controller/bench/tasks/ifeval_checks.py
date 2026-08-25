"""Programmatic instruction-following checkers for IFEval.

Vendored and adapted from Google Research's `instruction_following_eval`
(https://github.com/google-research/google-research/tree/master/instruction_following_eval,
Apache License 2.0) — the reference scorer for `google/IFEval`. Only
`check_following` logic is kept; `build_description` (which only matters for
generating NEW prompts, not scoring existing ones) is dropped, since every
row in the dataset already carries fully-resolved kwargs.

Two deliberate deviations from upstream, both to keep scoring fully offline
(no network / model download at score time, no LLM judge):
  - Sentence counting uses `_split_into_sentences` (upstream's own regex
    splitter, also vendored here) instead of `count_sentences`'s nltk
    `punkt` tokenizer, which needs `nltk.download` on first use.
  - Capital-word tokenization uses a regex approximation instead of nltk's
    `word_tokenize` (same `punkt`-download problem). Close enough for
    contractions/hyphenation; documented as an approximation, not exact
    parity with upstream on adversarial input.
Language detection (`language:response_language`, `change_case:english_*`)
uses `langdetect` (pinned in pyproject.toml) — its language profiles are
bundled in the wheel, so `langdetect.detect()` needs no network call.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Callable

import langdetect

_COMPARISON_RELATION = ("less than", "at least")

_CONSTRAINED_RESPONSE_OPTIONS = ("My answer is yes.", "My answer is no.", "My answer is maybe.")

# Sentence-boundary regex, vendored from instructions_util.py's
# split_into_sentences: a heuristic abbreviation/acronym-aware splitter that
# needs no downloaded model.
_ALPHABETS = "([A-Za-z])"
_PREFIXES = r"(Mr|St|Mrs|Ms|Dr)[.]"
_SUFFIXES = "(Inc|Ltd|Jr|Sr|Co)"
_STARTERS = (
    r"(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|"
    r"However\s|That\s|This\s|Wherever)"
)
_ACRONYMS = r"([A-Z][.][A-Z][.](?:[A-Z][.])?)"
_WEBSITES = "[.](com|net|org|io|gov|edu|me)"
_DIGITS = "([0-9])"
_MULTIPLE_DOTS = r"\.{2,}"


def _split_into_sentences(text: str) -> list[str]:
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(_PREFIXES, r"\1<prd>", text)
    text = re.sub(_WEBSITES, r"<prd>\1", text)
    text = re.sub(_DIGITS + "[.]" + _DIGITS, r"\1<prd>\2", text)
    text = re.sub(_MULTIPLE_DOTS, lambda m: "<prd>" * len(m.group(0)) + "<stop>", text)
    if "Ph.D" in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub(r"\s" + _ALPHABETS + r"[.] ", r" \1<prd> ", text)
    text = re.sub(_ACRONYMS + " " + _STARTERS, r"\1<stop> \2", text)
    triple_dot_pattern = _ALPHABETS + "[.]" + _ALPHABETS + "[.]" + _ALPHABETS + "[.]"
    text = re.sub(triple_dot_pattern, r"\1<prd>\2<prd>\3<prd>", text)
    text = re.sub(_ALPHABETS + "[.]" + _ALPHABETS + "[.]", r"\1<prd>\2<prd>", text)
    text = re.sub(" " + _SUFFIXES + "[.] " + _STARTERS, r" \1<stop> \2", text)
    text = re.sub(" " + _SUFFIXES + "[.]", r" \1<prd>", text)
    text = re.sub(" " + _ALPHABETS + "[.]", r" \1<prd>", text)
    if '"' in text:
        text = text.replace('."', '".')
    if "!" in text:
        text = text.replace('!"', '"!')
    if "?" in text:
        text = text.replace('?"', '"?')
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = [s.strip() for s in text.split("<stop>")]
    if sentences and not sentences[-1]:
        sentences = sentences[:-1]
    return sentences


def _count_words(text: str) -> int:
    return len(re.findall(r"\w+", text))


def _count_sentences(text: str) -> int:
    return len(_split_into_sentences(text))


def _capital_word_tokenize(text: str) -> list[str]:
    """Regex approximation of nltk's `word_tokenize`, hyphenated words as one token."""
    return re.findall(r"[A-Za-z]+(?:['’-][A-Za-z]+)*", text)


def _compare(actual: int, threshold: int, relation: str) -> bool:
    if relation == _COMPARISON_RELATION[0]:  # "less than"
        return actual < threshold
    return actual >= threshold  # "at least"


def _check_keyword_existence(response: str, kwargs: dict[str, Any]) -> bool:
    for keyword in kwargs["keywords"]:
        if not re.search(keyword, response, flags=re.IGNORECASE):
            return False
    return True


def _check_keyword_frequency(response: str, kwargs: dict[str, Any]) -> bool:
    count = len(re.findall(kwargs["keyword"], response, flags=re.IGNORECASE))
    return _compare(count, kwargs["frequency"], kwargs["relation"])


def _check_forbidden_words(response: str, kwargs: dict[str, Any]) -> bool:
    for word in kwargs["forbidden_words"]:
        if re.search(r"\b" + word + r"\b", response, flags=re.IGNORECASE):
            return False
    return True


def _check_letter_frequency(response: str, kwargs: dict[str, Any]) -> bool:
    letters = Counter(response.lower())
    count = letters[kwargs["letter"].lower()]
    return _compare(count, kwargs["let_frequency"], kwargs["let_relation"])


def _check_response_language(response: str, kwargs: dict[str, Any]) -> bool:
    try:
        return langdetect.detect(response) == kwargs["language"]
    except langdetect.LangDetectException:
        # Undetectable (e.g. too short/no linguistic content) counts as followed.
        return True


def _check_number_sentences(response: str, kwargs: dict[str, Any]) -> bool:
    return _compare(_count_sentences(response), kwargs["num_sentences"], kwargs["relation"])


def _check_number_paragraphs(response: str, kwargs: dict[str, Any]) -> bool:
    paragraphs = re.split(r"\s?\*\*\*\s?", response)
    num_paragraphs = len(paragraphs)
    for index, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            if index == 0 or index == len(paragraphs) - 1:
                num_paragraphs -= 1
            else:
                return False
    return num_paragraphs == kwargs["num_paragraphs"]


def _check_number_words(response: str, kwargs: dict[str, Any]) -> bool:
    return _compare(_count_words(response), kwargs["num_words"], kwargs["relation"])


def _check_nth_paragraph_first_word(response: str, kwargs: dict[str, Any]) -> bool:
    num_paragraphs = kwargs["num_paragraphs"]
    nth_paragraph = kwargs["nth_paragraph"]
    first_word = kwargs["first_word"].lower()

    paragraphs = re.split(r"\n\n", response)
    actual_num_paragraphs = len(paragraphs)
    for paragraph in paragraphs:
        if not paragraph.strip():
            actual_num_paragraphs -= 1

    # Faithful to upstream: the threshold check compares against the
    # blank-filtered count, but indexing still uses the unfiltered list.
    if nth_paragraph <= actual_num_paragraphs:
        paragraph = paragraphs[nth_paragraph - 1].strip()
        if not paragraph:
            return False
    else:
        return False

    word = paragraph.split()[0].strip().lstrip("'").lstrip('"')
    punctuation = {".", ",", "?", "!", "'", '"'}
    actual_first_word = ""
    for letter in word:
        if letter in punctuation:
            break
        actual_first_word += letter.lower()

    return actual_num_paragraphs == num_paragraphs and actual_first_word == first_word


def _check_number_placeholders(response: str, kwargs: dict[str, Any]) -> bool:
    return len(re.findall(r"\[.*?\]", response)) >= kwargs["num_placeholders"]


def _check_postscript(response: str, kwargs: dict[str, Any]) -> bool:
    marker = kwargs["postscript_marker"]
    value = response.lower()
    if marker == "P.P.S":
        pattern = r"\s*p\.\s?p\.\s?s.*$"
    elif marker == "P.S.":
        pattern = r"\s*p\.\s?s\..*$"
    else:
        pattern = r"\s*" + marker.lower() + r".*$"
    return bool(re.findall(pattern, value, flags=re.MULTILINE))


def _check_number_bullet_lists(response: str, kwargs: dict[str, Any]) -> bool:
    star_bullets = re.findall(r"^\s*\*[^\*].*$", response, flags=re.MULTILINE)
    dash_bullets = re.findall(r"^\s*-.*$", response, flags=re.MULTILINE)
    return len(star_bullets) + len(dash_bullets) == kwargs["num_bullets"]


def _check_constrained_response(response: str, kwargs: dict[str, Any]) -> bool:
    value = response.strip()
    return any(option in value for option in _CONSTRAINED_RESPONSE_OPTIONS)


def _check_number_highlighted_sections(response: str, kwargs: dict[str, Any]) -> bool:
    num_highlights = 0
    for highlight in re.findall(r"\*[^\n\*]*\*", response):
        if highlight.strip("*").strip():
            num_highlights += 1
    for highlight in re.findall(r"\*\*[^\n\*]*\*\*", response):
        if highlight.removeprefix("**").removesuffix("**").strip():
            num_highlights += 1
    return num_highlights >= kwargs["num_highlights"]


def _check_multiple_sections(response: str, kwargs: dict[str, Any]) -> bool:
    pattern = r"\s?" + kwargs["section_spliter"] + r"\s?\d+\s?"
    sections = re.split(pattern, response)
    return (len(sections) - 1) >= kwargs["num_sections"]


def _check_json_format(response: str, kwargs: dict[str, Any]) -> bool:
    value = (
        response.strip()
        .removeprefix("```json")
        .removeprefix("```Json")
        .removeprefix("```JSON")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    try:
        json.loads(value)
    except ValueError:
        return False
    return True


def _check_title(response: str, kwargs: dict[str, Any]) -> bool:
    for title in re.findall(r"<<[^\n]+>>", response):
        if title.lstrip("<").rstrip(">").strip():
            return True
    return False


def _check_two_responses(response: str, kwargs: dict[str, Any]) -> bool:
    parts = response.split("******")
    valid_responses = []
    for index, part in enumerate(parts):
        if not part.strip():
            if index != 0 and index != len(parts) - 1:
                return False
        else:
            valid_responses.append(part)
    return len(valid_responses) == 2 and valid_responses[0].strip() != valid_responses[1].strip()


def _check_repeat_prompt(response: str, kwargs: dict[str, Any]) -> bool:
    return response.strip().lower().startswith(kwargs["prompt_to_repeat"].strip().lower())


def _check_end_checker(response: str, kwargs: dict[str, Any]) -> bool:
    value = response.strip().strip('"').lower()
    end_phrase = kwargs["end_phrase"].strip().lower()
    return value.endswith(end_phrase)


def _check_capital_word_frequency(response: str, kwargs: dict[str, Any]) -> bool:
    words = _capital_word_tokenize(response)
    num_capital_words = sum(1 for w in words if w.isupper())
    return _compare(num_capital_words, kwargs["capital_frequency"], kwargs["capital_relation"])


def _check_english_capital(response: str, kwargs: dict[str, Any]) -> bool:
    try:
        return response.isupper() and langdetect.detect(response) == "en"
    except langdetect.LangDetectException:
        return True


def _check_english_lowercase(response: str, kwargs: dict[str, Any]) -> bool:
    try:
        return response.islower() and langdetect.detect(response) == "en"
    except langdetect.LangDetectException:
        return True


def _check_no_comma(response: str, kwargs: dict[str, Any]) -> bool:
    return "," not in response


def _check_quotation(response: str, kwargs: dict[str, Any]) -> bool:
    value = response.strip()
    return len(value) > 1 and value[0] == '"' and value[-1] == '"'


CHECKERS: dict[str, Callable[[str, dict[str, Any]], bool]] = {
    "keywords:existence": _check_keyword_existence,
    "keywords:frequency": _check_keyword_frequency,
    "keywords:forbidden_words": _check_forbidden_words,
    "keywords:letter_frequency": _check_letter_frequency,
    "language:response_language": _check_response_language,
    "length_constraints:number_sentences": _check_number_sentences,
    "length_constraints:number_paragraphs": _check_number_paragraphs,
    "length_constraints:number_words": _check_number_words,
    "length_constraints:nth_paragraph_first_word": _check_nth_paragraph_first_word,
    "detectable_content:number_placeholders": _check_number_placeholders,
    "detectable_content:postscript": _check_postscript,
    "detectable_format:number_bullet_lists": _check_number_bullet_lists,
    "detectable_format:constrained_response": _check_constrained_response,
    "detectable_format:number_highlighted_sections": _check_number_highlighted_sections,
    "detectable_format:multiple_sections": _check_multiple_sections,
    "detectable_format:json_format": _check_json_format,
    "detectable_format:title": _check_title,
    "combination:two_responses": _check_two_responses,
    "combination:repeat_prompt": _check_repeat_prompt,
    "startend:end_checker": _check_end_checker,
    "change_case:capital_word_frequency": _check_capital_word_frequency,
    "change_case:english_capital": _check_english_capital,
    "change_case:english_lowercase": _check_english_lowercase,
    "punctuation:no_comma": _check_no_comma,
    "startend:quotation": _check_quotation,
}


def check_instruction(instruction_id: str, response: str, kwargs: dict[str, Any]) -> bool:
    """Dispatch a single instruction check by id.

    Raises:
        KeyError: If `instruction_id` is not one of the 25 registered checkers.
    """
    return CHECKERS[instruction_id](response, kwargs)
