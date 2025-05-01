# utils.py
from nltk.corpus import words
from typing import Dict, List

LETTER_SCORES = {
    'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1,
    'F': 4, 'G': 2, 'H': 4, 'I': 1, 'J': 8,
    'K': 5, 'L': 1, 'M': 3, 'N': 1, 'O': 1,
    'P': 3, 'Q': 10, 'R': 1, 'S': 1, 'T': 1,
    'U': 1, 'V': 4, 'W': 4, 'X': 8, 'Y': 4,
    'Z': 10
}

# Load words from nltk
VALID_WORDS = set(word.upper() for word in words.words())


def is_valid_word(word: str) -> bool:
    """Check if word exists in dictionary"""
    return word.upper() in VALID_WORDS


def calculate_word_score(word: str, letter_multipliers: List[int] = None,
                         word_multiplier: int = 1) -> int:
    """Calculate score for a word with multipliers"""
    if letter_multipliers is None:
        letter_multipliers = [1] * len(word)

    score = 0
    for i, letter in enumerate(word.upper()):
        letter_score = LETTER_SCORES.get(letter, 0)
        score += letter_score * letter_multipliers[i]

    return score * word_multiplier


def generate_letter_multipliers(word: str, row: int, col: int,
                                direction: str, premium_squares: Dict) -> List[int]:
    """Generate letter multipliers based on premium squares"""
    multipliers = []
    for i in range(len(word)):
        r = row + (i if direction == "vertical" else 0)
        c = col + (i if direction == "horizontal" else 0)
        square = premium_squares.get((r, c))
        if square == 'DL':
            multipliers.append(2)
        elif square == 'TL':
            multipliers.append(3)
        else:
            multipliers.append(1)
    return multipliers
