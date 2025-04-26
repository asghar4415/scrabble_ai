# game_logic/scrabble_utils.py

from nltk.corpus import words

LETTER_SCORES = {
    'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1,
    'F': 4, 'G': 2, 'H': 4, 'I': 1, 'J': 8,
    'K': 5, 'L': 1, 'M': 3, 'N': 1, 'O': 1,
    'P': 3, 'Q': 10,'R': 1, 'S': 1, 'T': 1,
    'U': 1, 'V': 4, 'W': 4, 'X': 8, 'Y': 4,
    'Z': 10
}

# Load words from nltk
VALID_WORDS = set(word.upper() for word in words.words())

def is_valid_word(word: str) -> bool:
    return word.upper() in VALID_WORDS

def calculate_word_score(word, letter_multipliers=None, word_multiplier=1):
    score = 0
    for i, letter in enumerate(word):
        letter_score = LETTER_SCORES.get(letter.upper(), 0)
        multiplier = letter_multipliers[i] if letter_multipliers else 1
        score += letter_score * multiplier
    return score * word_multiplier
