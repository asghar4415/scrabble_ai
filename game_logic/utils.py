
import logging
import os
from typing import Set, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # Avoid circular import for type hinting
    from .board import Board


logger = logging.getLogger(__name__)

VALID_WORDS: Set[str] = set()  # Will be populated by initialize_dictionary


def initialize_dictionary() -> Set[str]:
    global VALID_WORDS  # Allow modification of the global VALID_WORDS
    words = set()
    possible_paths = [
        'scrabble_words.txt',
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
            __file__))), 'scrabble_words.txt'),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(
            __file__)))), 'scrabble_words.txt')
    ]
    dict_path_found = next(
        (path for path in possible_paths if os.path.exists(path)), None)

    minimal_word_set = {"QI", "ZA", "CAT", "DOG", "JO", "AX", "EX",
                        "OX", "XI", "XU", "WORD", "PLAY", "GAME", "POWER", "TURN"}

    if not dict_path_found:
        logger.warning("scrabble_words.txt not found. Using minimal word set.")
        VALID_WORDS = minimal_word_set
        return minimal_word_set
    try:
        with open(dict_path_found, 'r', encoding='utf-8') as f:
            loaded_words = {
                line.strip().upper() for line in f
                if len(line.strip()) >= 2 and line.strip().isalpha()
            }
        if not loaded_words:
            logger.warning(
                f"Dictionary file {dict_path_found} was empty or contained no valid words. Using minimal set.")
            VALID_WORDS = minimal_word_set
        else:
            VALID_WORDS = loaded_words
            logger.info(
                f"Successfully loaded {len(VALID_WORDS)} words from {dict_path_found}")
        return VALID_WORDS
    except Exception as e:
        logger.error(
            f"Error reading dictionary file {dict_path_found}: {e}. Using minimal word set.")
        VALID_WORDS = minimal_word_set
        return minimal_word_set


def is_valid_word(word: str) -> bool:
    """Checks if a word is valid according to the loaded dictionary and basic rules."""
    if not VALID_WORDS:  # Ensure dictionary is loaded
        initialize_dictionary()
    if not word or len(word) < 2 or not word.isalpha():
        return False
    return word.upper() in VALID_WORDS


def find_anchor_squares(board: 'Board') -> List[Tuple[int, int]]:
    """Identifies potential squares on the board where new tiles can be placed adjacent to existing ones."""
    center_square = (board.size // 2, board.size // 2)
    if board.first_move:
        return [center_square]

    anchors = set()
    for r in range(board.size):
        for c in range(board.size):
            if board.get_raw_tile_marker_at(r, c) is None:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    adj_r, adj_c = r + dr, c + dc
                    if 0 <= adj_r < board.size and 0 <= adj_c < board.size and \
                       board.get_raw_tile_marker_at(adj_r, adj_c) is not None:
                        anchors.add((r, c))
                        break
    return list(anchors) if anchors else [center_square]


initialize_dictionary()
