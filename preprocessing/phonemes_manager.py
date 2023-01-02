from collections import defaultdict
from dataclasses import dataclass
from typing import Dict


@dataclass
class PhonemeFamily:
    """
    Class representing a family of phonemes.
    """
    name: str
    ui_name: str
    color: str


@dataclass
class Phoneme:
    """
    Class representing a phoneme.
    """
    symbol: str
    family: PhonemeFamily
    color: str
    index: int


# Phoneme families
stops = PhonemeFamily('stops', 'ST', '#f75b40')
affricates = PhonemeFamily('affricates', 'AF', '#FFD700')
fricatives = PhonemeFamily('fricatives', 'FR', '#90d18d')
nasals = PhonemeFamily('nasals', 'NA', '#7a7a7a')
semivowels = PhonemeFamily('semivowels', 'SV', '#8683bd')
vowels = PhonemeFamily('vowels', 'V', '#529dcc')
others = PhonemeFamily('others', 'O', '#CD853F')

PHONEMES = {
    'b': Phoneme('b', stops, '#fdcbb6', 0),
    'd': Phoneme('d', stops, '#fca78b', 1),
    'g': Phoneme('g', stops, '#fc8262', 2),
    'p': Phoneme('p', stops, '#f75b40', 3),
    't': Phoneme('t', stops, '#e63328', 4),
    'k': Phoneme('k', stops, '#c4161c', 5),
    'dx': Phoneme('dx', stops, '#9f0e14', 6),
    'q': Phoneme('q', stops, '#67000d', 7),

    'jh': Phoneme('b', affricates, '#FFD700', 8),
    'ch': Phoneme('b', affricates, '#FFD700', 9),

    's': Phoneme('s', fricatives, '#d4eece', 10),
    'sh': Phoneme('sh', fricatives, '#b5e1ae', 11),
    'z': Phoneme('z', fricatives, '#90d18d', 12),
    'zh': Phoneme('zh', fricatives, '#90d18d', 13),
    'f': Phoneme('f', fricatives, '#90d18d', 14),
    'th': Phoneme('th', fricatives, '#1d8640', 15),
    'v': Phoneme('v', fricatives, '#00692a', 16),
    'dh': Phoneme('dh', fricatives, '#00441b', 17),

    'm': Phoneme('m', nasals, '#e3e3e3', 18),
    'n': Phoneme('n', nasals, '#c7c7c7', 19),
    'ng': Phoneme('ng', nasals, '#a2a2a2', 20),
    'em': Phoneme('em', nasals, '#7a7a7a', 21),
    'en': Phoneme('en', nasals, '#565656', 22),
    'eng': Phoneme('eng', nasals, '#282828', 23),
    'nx': Phoneme('nx', nasals, '#1d1d1d', 24),

    'l': Phoneme('l', semivowels, '#e3e2ef', 25),
    'r': Phoneme('r', semivowels, '#c7c8e1', 26),
    'w': Phoneme('w', semivowels, '#a7a4ce', 27),
    'y': Phoneme('y', semivowels, '#8683bd', 28),
    'hh': Phoneme('hh', semivowels, '#6d57a6', 29),
    'hv': Phoneme('hv', semivowels, '#552a90', 30),
    'el': Phoneme('el', semivowels, '#3f007d', 31),

    'iy': Phoneme('iy', vowels, '#d0e2f2', 32),
    'ih': Phoneme('ih', vowels, '#c9ddf0', 33),
    'eh': Phoneme('eh', vowels, '#bdd7ec', 34),
    'ey': Phoneme('ey', vowels, '#afd1e7', 35),
    'ae': Phoneme('ae', vowels, '#a1cbe2', 36),
    'aa': Phoneme('aa', vowels, '#91c3de', 37),
    'aw': Phoneme('aw', vowels, '#7fb9da', 38),
    'ay': Phoneme('ay', vowels, '#6fb0d7', 39),
    'ah': Phoneme('ah', vowels, '#60a7d2', 40),
    'ao': Phoneme('ao', vowels, '#529dcc', 41),
    'oy': Phoneme('oy', vowels, '#4493c7', 42),
    'ow': Phoneme('ow', vowels, '#3888c1', 43),
    'uh': Phoneme('uh', vowels, '#2d7dbb', 44),
    'uw': Phoneme('uw', vowels, '#2272b6', 45),
    'ux': Phoneme('ux', vowels, '#1967ad', 46),
    'er': Phoneme('er', vowels, '#115ca5', 47),
    'ax': Phoneme('ax', vowels, '#08519c', 48),
    'ix': Phoneme('ix', vowels, '#08468b', 49),
    'axr': Phoneme('axr', vowels, '#083a7a', 50),
    'ax-h': Phoneme('ax-h', vowels, '#08306b', 51),

    'pau': Phoneme('pau', others, '#F4A460', 52),
    'epi': Phoneme('epi', others, "#F4A460", 53),
    'h#': Phoneme('#h', others, '#CD853F', 54),
    '1': Phoneme('1', others, '#CD853F', 55),
    '2': Phoneme('2', others, '#CD853F', 56),

    '?': Phoneme('?', others, "#000000", 57),

}


def get_phones_families():
    """
    Returns a list of all phoneme families.
    """
    return [stops, affricates, fricatives, nasals, semivowels, vowels, others]


def phone_name_to_family_color(name: str) -> str:
    """
    Returns the color of the phoneme family that the given phoneme belongs to.
    If the phoneme is not in the list of known phonemes, returns the color of the phoneme family that the first character
    of the phoneme belongs to, if it is in the list of known phonemes. If it is not, returns black.
    """
    if name not in PHONEMES:
        if name[0] in PHONEMES:
            return PHONEMES[name[0]].family.color
        return "#000000"
    return PHONEMES[name].family.color


def phone_name_to_family_name(phone_name: str) -> str:
    """
    Given a phone name (a string), returns the corresponding family name (also a string).
    If the given phone name is not in the PHONEMES dictionary, the function checks if the
    first character of the phone name is in the PHONEMES dictionary. If it is, the function
    returns the family name of the phoneme corresponding to that character. If the phone name
    is not in the PHONEMES dictionary and the first character is also not in the PHONEMES
    dictionary, the function returns the string "UN".
    """
    if phone_name not in PHONEMES:
        if phone_name[0] in PHONEMES:
            return PHONEMES[phone_name[0]].family.ui_name
        return "UN"
    return PHONEMES[phone_name].family.ui_name


def phone_txt_to_family_txt(phone_text: str) -> str:
    """
    Given a string of phone text, returns a string of the corresponding family names.
    If the given phone text is the string "?", the function returns "?".
    Otherwise, the function splits the phone text into lines and processes each line separately.
    For each line, the function splits the line into a phone name and a score, converts the score
    to an integer, and gets the family name corresponding to the phone name using the
    phone_name_to_family_name() function. The function keeps a tally of the scores for each
    family name in a dictionary, and at the end, it constructs the final text string by
    concatenating the family names and scores for each non-zero score in the dictionary.
    """
    final_text = ""
    if phone_text == "?":
        return "?"
    family_name_to_score: Dict[str, int] = defaultdict(int)
    for phone_line in phone_text.splitlines():
        phone_name, phone_score = phone_line.split("(")
        score = int(phone_score.replace("%)", ""))
        family_name = phone_name_to_family_name(phone_name)
        family_name_to_score[family_name] += score
    for name, score in family_name_to_score.items():
        if score == 0:
            continue
        final_text += f'{name} ({score}%)'
    return final_text


from typing import List


def phone_name_to_color(name: str) -> str:
    """
    Given a phone name, returns the corresponding color.
    If the phone name is not found in the list of phonemes,
    returns the color of the phoneme corresponding to the first
    letter of the phone name, if it exists. If it does not exist,
    returns the color of the '?' phoneme.

    Parameters:
    - name: str - the name of the phone

    Returns:
    - str - the color of the phone
    """
    name = name.lower()
    if name not in PHONEMES:
        if name[0] in PHONEMES:
            return PHONEMES[name[0]].color
        return PHONEMES["?"].color
    return PHONEMES[name].color


def get_phonemes_in_family(family: PhonemeFamily) -> List[Phoneme]:
    """
    Given a phoneme family, returns a list of phonemes belonging to that family.
    """
    return [p for p in PHONEMES.values() if p.family == family]


def phoneme_to_index(name: str):
    """
    Given a phoneme name, returns the corresponding index.
    If the phoneme name is not found in the list of phonemes,
    returns the index of the phoneme corresponding to the first
    letter of the phoneme name, if it exists. If it does not exist,
    returns the index of the '?' phoneme.
    """
    name = name.lower()
    if name not in PHONEMES:
        if name[0] in PHONEMES:
            return PHONEMES[name[0]].index
        return PHONEMES["?"].index
    return PHONEMES[name].index


def index_to_phoneme(index: int):
    """
    Given an index, returns the corresponding phoneme name.
    If no phoneme is found with the given index, returns '?'.
    """
    for p in PHONEMES:
        if PHONEMES[p].index == index:
            return p
    return "?"
