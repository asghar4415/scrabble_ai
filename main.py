# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Tuple, Set, Any
import random
from copy import deepcopy
import logging
import time
import json
from itertools import permutations
import os

# --------------------------
# Initialization and Configuration
# --------------------------

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Scrabble AI Game Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173",
                   "http://127.0.0.1:5173", "https://scrabble-sandy.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# --------------------------
# Game Constants and Utilities
# --------------------------

LETTER_SCORES = {
    'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1,
    'F': 4, 'G': 2, 'H': 4, 'I': 1, 'J': 8,
    'K': 5, 'L': 1, 'M': 3, 'N': 1, 'O': 1,
    'P': 3, 'Q': 10, 'R': 1, 'S': 1, 'T': 1,
    'U': 1, 'V': 4, 'W': 4, 'X': 8, 'Y': 4,
    'Z': 10, ' ': 0
}


def initialize_dictionary() -> Set[str]:
    """
    Loads valid Scrabble words from a file into a set for efficient lookup.
    Looks for 'scrabble_words.txt' in common locations. If not found or empty,
    it falls back to a minimal set of words for basic functionality.
    """
    words = set()
    possible_paths = [
        'scrabble_words.txt',
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     'scrabble_words.txt'),
        './scrabble_words.txt'
    ]
    dict_path_found = None
    for path in possible_paths:
        if os.path.exists(path):
            dict_path_found = path
            break

    minimal_word_set = {"QI", "ZA", "CAT", "DOG", "JO",
                        "AX", "EX", "OX", "XI", "XU", "WORD", "PLAY", "GAME"}

    if not dict_path_found:
        logger.error(
            "scrabble_words.txt not found in expected locations. Using minimal word set.")
        return minimal_word_set

    try:
        with open(dict_path_found, 'r', encoding='utf-8') as f:
            loaded_words = {line.strip().upper() for line in f if len(
                line.strip()) >= 2 and line.strip().isalpha()}

            if not loaded_words:
                logger.warning(
                    f"{dict_path_found} seems empty or contains no valid words, using minimal set.")
                words = minimal_word_set
            else:
                words = loaded_words
                logger.info(
                    f"Loaded {len(words)} words from {dict_path_found}")
            return words
    except Exception as e:
        logger.error(
            f"Error reading {dict_path_found}: {e}. Using minimal word set.")
        return minimal_word_set


VALID_WORDS = initialize_dictionary()


def is_valid_word(word: str) -> bool:
    """
    Checks if a given word is present in the loaded Scrabble dictionary.
    """
    if not word or len(word) < 2 or not word.isalpha():
        return False
    return word.upper() in VALID_WORDS

# --------------------------
# Game Board Class
# --------------------------


class Board:
    """
    Represents the Scrabble game board, state, and core game logic.
    Manages the board grid, tile bag, player racks, scores, turns,
    premium squares, move validation, scoring, and game end conditions.
    """

    def __init__(self):
        """Initializes a new Scrabble game board and state."""
        self.size = 15
        self.board: List[List[Optional[Tuple[str, bool]]]] = [[None for _ in range(self.size)]
                                                              for _ in range(self.size)]
        self.scores = {"human": 0, "ai": 0}
        self.tile_bag = self.initialize_tile_bag()
        self.player_racks = {
            "human": self.draw_tiles(7),
            "ai": self.draw_tiles(7)
        }
        self.current_player = "human"
        self.game_over = False
        self.first_move = True
        self.premium_squares = self.initialize_premium_squares()
        self.last_move_time = time.time()
        self.consecutive_passes = 0

    def get_letter_at(self, r: int, c: int) -> Optional[str]:
        """
        Retrieves the letter placed at a specific board coordinate.
        Returns the letter or None if empty/out of bounds.
        """
        if 0 <= r < self.size and 0 <= c < self.size and self.board[r][c]:
            return self.board[r][c][0]
        return None

    def initialize_tile_bag(self) -> List[str]:
        """
        Creates and shuffles the tile bag with the standard Scrabble letter distribution.
        """
        distribution = {
            'A': 9, 'B': 2, 'C': 2, 'D': 4, 'E': 12, 'F': 2, 'G': 3, 'H': 2,
            'I': 9, 'J': 1, 'K': 1, 'L': 4, 'M': 2, 'N': 6, 'O': 8, 'P': 2,
            'Q': 1, 'R': 6, 'S': 4, 'T': 6, 'U': 4, 'V': 2, 'W': 2, 'X': 1,
            'Y': 2, 'Z': 1, ' ': 2
        }
        tile_bag = []
        for letter, count in distribution.items():
            tile_bag.extend([letter] * count)
        random.shuffle(tile_bag)
        logger.info(f"Initialized tile bag with {len(tile_bag)} tiles.")
        return tile_bag

    def draw_tiles(self, count: int) -> List[str]:
        """
        Draws a specified number of tiles from the tile bag.
        Returns fewer if the bag runs out.
        """
        drawn = []
        for _ in range(count):
            if not self.tile_bag:
                break
            drawn.append(self.tile_bag.pop())
        return drawn

    def initialize_premium_squares(self) -> Dict[Tuple[int, int], str]:
        """
        Sets up the locations and types of premium squares (TW, DW, TL, DL) on the board.
        """
        premiums = {}
        size = self.size
        center = size // 2

        tw_coords = [(0, 0), (0, 7), (7, 0)]
        dw_coords = [(r, r) for r in range(1, 5)] + [(center, center)] + \
                    [(r, size - 1 - r) for r in range(1, 5)]
        tl_coords = [(1, 5), (1, 9), (5, 1), (5, 5), (5, 9), (5, 13)]
        dl_coords = [(0, 3), (0, 11), (2, 6), (2, 8), (3, 0), (3, 7), (3, 14),
                     (6, 2), (6, 6), (6, 8), (6, 12), (7, 3), (7, 11)]

        all_coords = set((r, c) for r in range(size) for c in range(size))

        def mirror(r, c):
            return [(r, c), (r, size-1-c), (size-1-r, c), (size-1-r, size-1-c)]

        for r, c in tw_coords:
            for mr, mc in mirror(r, c):
                if (mr, mc) in all_coords:
                    premiums[(mr, mc)] = 'TW'
        for r, c in dw_coords:
            for mr, mc in mirror(r, c):
                if (mr, mc) in all_coords:
                    premiums[(mr, mc)] = 'DW'
        for r, c in tl_coords:
            for mr, mc in mirror(r, c):
                if (mr, mc) in all_coords:
                    premiums[(mr, mc)] = 'TL'
        for r, c in dl_coords:
            for mr, mc in mirror(r, c):
                if (mr, mc) in all_coords:
                    premiums[(mr, mc)] = 'DL'

        premiums[(center, center)] = 'DW'
        logger.info(f"Initialized {len(premiums)} premium squares.")
        return premiums

    def calculate_move_score(self, word: str, row: int, col: int, direction: str,
                             placed_tiles_info: List[Dict[str, Any]]) -> Tuple[int, List[str]]:
        """
        Calculates the total score for a valid move, including the main word and any crosswords formed.
        Accounts for premium squares (only applied by newly placed tiles) and the bingo bonus.
        """
        total_score = 0
        words_formed = []
        main_word_score = 0
        main_word_multiplier = 1
        placed_positions = {info['pos'] for info in placed_tiles_info}
        placed_letters_map = {info['pos']: info['letter']
                              for info in placed_tiles_info}

        logger.debug(
            f"Scoring main word: '{word}' at ({row},{col}) {direction}")
        for i, letter in enumerate(word.upper()):
            r = row + (i if direction == "vertical" else 0)
            c = col + (i if direction == "horizontal" else 0)
            current_pos = (r, c)
            square_type = self.premium_squares.get(current_pos)
            is_newly_placed = current_pos in placed_positions

            letter_value = LETTER_SCORES.get(letter, 0)
            letter_multiplier = 1

            if is_newly_placed:
                is_blank = any(info['pos'] == current_pos and info['is_blank']
                               for info in placed_tiles_info)
                if is_blank:
                    letter_value = 0

                logger.debug(
                    f"  Tile {letter} at {current_pos} is new. Premium: {square_type}")
                if square_type == 'DL':
                    letter_multiplier = 2
                elif square_type == 'TL':
                    letter_multiplier = 3
                elif square_type == 'DW':
                    main_word_multiplier *= 2
                elif square_type == 'TW':
                    main_word_multiplier *= 3
            else:
                logger.debug(f"  Tile {letter} at {current_pos} is existing.")

            main_word_score += letter_value * letter_multiplier
            logger.debug(
                f"    Letter score: {letter_value} * {letter_multiplier}. Running word score: {main_word_score}")

        main_word_score *= main_word_multiplier
        total_score += main_word_score
        words_formed.append(word)
        logger.debug(
            f"  Main word '{word}' base: {main_word_score // main_word_multiplier}, Multiplier: x{main_word_multiplier}, Final: {main_word_score}")

        cross_direction = "vertical" if direction == "horizontal" else "horizontal"
        logger.debug("Checking for crosswords:")
        for info in placed_tiles_info:
            r_placed, c_placed = info['pos']
            placed_letter = info['letter']
            is_blank_placed = info['is_blank']
            logger.debug(
                f"  Checking cross potential for '{placed_letter}' at ({r_placed},{c_placed})")

            cross_word_list = [placed_letter]
            cross_word_start_pos = (r_placed, c_placed)

            cr, cc = r_placed, c_placed
            while True:
                nr, nc = (
                    cr - 1, cc) if cross_direction == "vertical" else (cr, cc - 1)
                if not (0 <= nr < self.size and 0 <= nc < self.size):
                    break
                existing_letter = self.get_letter_at(nr, nc)
                newly_placed_in_cross = placed_letters_map.get(
                    (nr, nc)) if (nr, nc) in placed_positions else None
                letter_to_add = existing_letter or newly_placed_in_cross
                if letter_to_add:
                    cross_word_list.insert(0, letter_to_add)
                    cross_word_start_pos = (nr, nc)
                    cr, cc = nr, nc
                else:
                    break

            cr, cc = r_placed, c_placed
            while True:
                nr, nc = (
                    cr + 1, cc) if cross_direction == "vertical" else (cr, cc + 1)
                if not (0 <= nr < self.size and 0 <= nc < self.size):
                    break
                existing_letter = self.get_letter_at(nr, nc)
                newly_placed_in_cross = placed_letters_map.get(
                    (nr, nc)) if (nr, nc) in placed_positions else None
                letter_to_add = existing_letter or newly_placed_in_cross
                if letter_to_add:
                    cross_word_list.append(letter_to_add)
                    cr, cc = nr, nc
                else:
                    break

            cross_word = "".join(cross_word_list)
            logger.debug(
                f"    Potential cross word: '{cross_word}' starting at {cross_word_start_pos}")

            if len(cross_word) >= 2 and is_valid_word(cross_word):
                if cross_word not in words_formed:
                    words_formed.append(cross_word)
                    cross_score = 0
                    cross_word_multiplier = 1
                    logger.debug(
                        f"    Scoring valid cross word '{cross_word}'")

                    for i, letter in enumerate(cross_word.upper()):
                        r_cross = cross_word_start_pos[0] + \
                            (i if cross_direction == "vertical" else 0)
                        c_cross = cross_word_start_pos[1] + \
                            (i if cross_direction == "horizontal" else 0)
                        current_pos = (r_cross, c_cross)
                        square_type = self.premium_squares.get(current_pos)

                        letter_value = LETTER_SCORES.get(letter, 0)
                        is_newly_placed_blank_here = False
                        if current_pos == (r_placed, c_placed):
                            is_newly_placed_blank_here = is_blank_placed
                        elif current_pos in placed_positions:
                            is_newly_placed_blank_here = any(
                                p_info['pos'] == current_pos and p_info['is_blank'] for p_info in placed_tiles_info)

                        if is_newly_placed_blank_here:
                            letter_value = 0

                        letter_multiplier = 1
                        if current_pos == (r_placed, c_placed):
                            logger.debug(
                                f"      Anchor {letter} at {current_pos}. Premium: {square_type}")
                            if square_type == 'DL':
                                letter_multiplier = 2
                            elif square_type == 'TL':
                                letter_multiplier = 3
                            if square_type == 'DW':
                                cross_word_multiplier *= 2
                            elif square_type == 'TW':
                                cross_word_multiplier *= 3
                        else:
                            logger.debug(
                                f"      Non-anchor {letter} at {current_pos}.")

                        cross_score += letter_value * letter_multiplier
                        logger.debug(
                            f"        Letter score: {letter_value} * {letter_multiplier}. Running cross score: {cross_score}")

                    cross_score *= cross_word_multiplier
                    total_score += cross_score
                    logger.debug(
                        f"      Cross word '{cross_word}' base: {cross_score // cross_word_multiplier}, Multiplier: x{cross_word_multiplier}, Final: {cross_score}")
                else:
                    logger.debug(
                        f"    Cross word '{cross_word}' already counted (likely same as main word).")
            elif len(cross_word) >= 2:
                logger.warning(
                    f"    Invalid crossword '{cross_word}' detected during scoring at ({r_placed},{c_placed}). Validation might have missed this.")
            else:
                logger.debug(
                    f"    Sequence '{cross_word}' too short to be a word.")

        if len(placed_positions) == 7:
            logger.info("Bingo! Player used all 7 tiles. +50 points.")
            total_score += 50

        unique_words_formed = sorted(list(set(words_formed)))
        logger.info(
            f"Total score for move: {total_score}. Words formed: {unique_words_formed}")
        return total_score, unique_words_formed

    def validate_move(self, word_proposal: str, row: int, col: int, direction: str, player: str) -> Tuple[bool, str, List[Dict[str, Any]]]:
        """
        Performs comprehensive validation of a potential move before placement.
        Checks boundaries, overlaps, rack tiles (incl. blanks), game rules, and word validity (main + crosswords).
        Returns (isValid, message, placedTilesInfo).
        """
        word_proposal = word_proposal.upper()
        rack = self.player_racks[player]
        temp_board = deepcopy(self.board)
        placed_tiles_info = []
        letters_needed_from_rack = []
        connected_to_existing = self.first_move
        center_square = (self.size // 2, self.size // 2)
        touches_center = False
        num_new_tiles = 0

        if not word_proposal or not all(c.isalpha() or c == ' ' for c in word_proposal):
            return False, "Word cannot be empty and must contain only letters or spaces (for blanks).", []

        word_len = len(word_proposal)
        current_word_parts = []

        if direction == "horizontal":
            if col < 0 or col + word_len > self.size or row < 0 or row >= self.size:
                return False, "Word placement out of bounds.", []
        else:
            if row < 0 or row + word_len > self.size or col < 0 or col >= self.size:
                return False, "Word placement out of bounds.", []

        for i, proposed_letter in enumerate(word_proposal):
            r = row + (i if direction == "vertical" else 0)
            c = col + (i if direction == "horizontal" else 0)
            pos = (r, c)
            existing_tile_info = self.board[r][c]

            if existing_tile_info:
                existing_letter = existing_tile_info[0]
                if existing_letter != proposed_letter:
                    return False, f"Placement conflict at ({r},{c}): Board has '{existing_letter}', proposed '{proposed_letter}'.", []
                current_word_parts.append(existing_letter)
                connected_to_existing = True
            else:
                num_new_tiles += 1
                current_word_parts.append(proposed_letter)
                letters_needed_from_rack.append(proposed_letter)
                placed_tiles_info.append(
                    {'letter': proposed_letter, 'pos': pos, 'is_blank': False})

                if not connected_to_existing:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.size and 0 <= nc < self.size and self.board[nr][nc]:
                            connected_to_existing = True
                            break

            if self.first_move and pos == center_square:
                touches_center = True
            temp_board[r][c] = (proposed_letter, False)

        if num_new_tiles == 0:
            return False, "Move must place at least one new tile.", []

        formed_word = "".join(current_word_parts)

        rack_copy = list(rack)
        final_placed_tiles_info = []
        possible = True
        temp_placed_info_copy = list(placed_tiles_info)

        for letter_needed in letters_needed_from_rack:
            if letter_needed in rack_copy:
                rack_copy.remove(letter_needed)
                for info in temp_placed_info_copy:
                    if info['letter'] == letter_needed and info not in final_placed_tiles_info:
                        info['is_blank'] = False
                        final_placed_tiles_info.append(info)
                        temp_placed_info_copy.remove(info)
                        break
            elif ' ' in rack_copy:
                rack_copy.remove(' ')
                for info in temp_placed_info_copy:
                    if info['letter'] == letter_needed and info not in final_placed_tiles_info:
                        info['is_blank'] = True
                        final_placed_tiles_info.append(info)
                        temp_placed_info_copy.remove(info)
                        break
            else:
                possible = False
                break

        if not possible:
            return False, f"Letter '{letter_needed}' not available in rack {rack} (considering blanks).", []

        if len(final_placed_tiles_info) != len(letters_needed_from_rack):
            logger.error(
                f"Internal error during rack validation. Need: {letters_needed_from_rack}, Found: {final_placed_tiles_info}")
            return False, "Internal error during rack validation.", []

        placed_tiles_info = final_placed_tiles_info

        if self.first_move and not touches_center:
            return False, "First move must touch the center square.", []
        if not self.first_move and not connected_to_existing:
            return False, "Word must connect to existing tiles (adjacent or overlap).", []

        if not is_valid_word(formed_word):
            return False, f"Main word '{formed_word}' is not in the dictionary.", []

        cross_direction = "vertical" if direction == "horizontal" else "horizontal"
        for info in placed_tiles_info:
            r_placed, c_placed = info['pos']
            placed_letter = info['letter']
            cross_word_list = [placed_letter]

            cr, cc = r_placed, c_placed
            while True:
                nr, nc = (
                    cr - 1, cc) if cross_direction == "vertical" else (cr, cc - 1)
                if not (0 <= nr < self.size and 0 <= nc < self.size):
                    break
                tile_info = temp_board[nr][nc]
                if tile_info:
                    cross_word_list.insert(0, tile_info[0])
                    cr, cc = nr, nc
                else:
                    break

            cr, cc = r_placed, c_placed
            while True:
                nr, nc = (
                    cr + 1, cc) if cross_direction == "vertical" else (cr, cc + 1)
                if not (0 <= nr < self.size and 0 <= nc < self.size):
                    break
                tile_info = temp_board[nr][nc]
                if tile_info:
                    cross_word_list.append(tile_info[0])
                    cr, cc = nr, nc
                else:
                    break

            cross_word = "".join(cross_word_list)
            if len(cross_word) >= 2 and not is_valid_word(cross_word):
                return False, f"Placement creates invalid crossword '{cross_word}' at ({r_placed},{c_placed}).", []

        return True, f"Move '{formed_word}' is valid.", placed_tiles_info

    def place_word(self, word: str, row: int, col: int, direction: str, player: str) -> Tuple[bool, str, Optional[int]]:
        """
        Attempts to validate and place a word on the board, updating game state if valid.
        Handles scoring, board update, rack management, state flags, game end checks, and turn switching.
        Returns (success, message, score).
        """
        logger.info(
            f"Attempting move by {player}: word='{word}', pos=({row},{col}), dir={direction}")
        is_valid, message, placed_tiles_info = self.validate_move(
            word, row, col, direction, player)

        if not is_valid:
            logger.warning(f"Validation failed for {player}: {message}")
            return False, message, None

        formed_word = AIPlayer._get_formed_word(
            self, word, row, col, direction, placed_tiles_info)
        logger.info(f"Validated formed word: '{formed_word}'")

        start_r_score, start_c_score = row, col
        if direction == "horizontal":
            while start_c_score > 0 and self.get_letter_at(row, start_c_score - 1):
                start_c_score -= 1
        else:
            while start_r_score > 0 and self.get_letter_at(start_r_score - 1, col):
                start_r_score -= 1

        score, words_formed = self.calculate_move_score(
            formed_word, start_r_score, start_c_score, direction, placed_tiles_info
        )
        logger.info(
            f"Player {player} formed words: {words_formed} for score: {score}")

        letters_removed_from_rack = []
        for info in placed_tiles_info:
            r_place, c_place = info['pos']
            letter_placed = info['letter']
            self.board[r_place][c_place] = (letter_placed, True)
            letters_removed_from_rack.append(
                ' ' if info['is_blank'] else letter_placed)

        self.scores[player] += score
        self.remove_from_rack(letters_removed_from_rack, player)
        self.refill_rack(player)

        self.first_move = False
        self.consecutive_passes = 0
        self.last_move_time = time.time()

        if not self.tile_bag and not self.player_racks[player]:
            self.game_over = True
            self.finalize_scores()
            logger.info(
                f"Game Over: Player {player} used last tile and bag is empty.")

        if not self.game_over:
            self.switch_turn()

        success_message = f"Played '{formed_word}' for {score} points. Words formed: {', '.join(words_formed)}."
        return True, success_message, score

    def pass_turn(self, player: str) -> Tuple[bool, str]:
        """
        Allows the current player to pass their turn. Handles consecutive passes and game end condition.
        """
        if player != self.current_player:
            logger.warning(
                f"Pass attempt by {player} but it's {self.current_player}'s turn.")
            return False, "Not your turn to pass."

        self.consecutive_passes += 1
        logger.info(
            f"Player {player} passed. Consecutive passes: {self.consecutive_passes}")

        if self.consecutive_passes >= 6:
            self.game_over = True
            self.finalize_scores()
            logger.info("Game Over: 6 consecutive passes.")
            return True, "Turn passed. Game Over due to 6 consecutive passes."
        else:
            self.switch_turn()
            return True, "Turn passed."

    def finalize_scores(self):
        """
        Adjusts final scores at the end of the game based on unplayed tiles.
        Deducts points for remaining tiles and awards them to the player who finished, if any.
        """
        logger.info("Game Over - Finalizing Scores.")
        empty_rack_player = None
        total_unplayed_score = 0

        for p, rack in self.player_racks.items():
            player_unplayed_score = sum(
                LETTER_SCORES.get(tile, 0) for tile in rack)
            if not rack:
                empty_rack_player = p
                logger.info(f"Player {p} finished with an empty rack.")
            else:
                self.scores[p] -= player_unplayed_score
                total_unplayed_score += player_unplayed_score
                logger.info(
                    f"Player {p} deducts {player_unplayed_score} points for unplayed tiles: {rack}")

        if empty_rack_player and self.get_opponent(empty_rack_player) in self.scores:
            self.scores[empty_rack_player] += total_unplayed_score
            logger.info(
                f"Player {empty_rack_player} receives {total_unplayed_score} points from opponent's unplayed tiles.")
        elif empty_rack_player:
            logger.warning(
                f"Score finalization: player {empty_rack_player} has empty rack, but opponent score not found.")
        else:
            logger.info(
                "Game ended with tiles remaining for both players (likely due to passes). No points transferred.")

        logger.info(
            f"Final Scores: Human: {self.scores.get('human', 0)}, AI: {self.scores.get('ai', 0)}")

    def remove_from_rack(self, letters: List[str], player: str):
        """
        Removes a list of specified letters (including blanks as ' ') from a player's rack.
        """
        rack = self.player_racks[player]
        logger.debug(f"Removing {letters} from {player}'s rack {rack}")
        for letter in letters:
            try:
                rack.remove(letter)
            except ValueError:
                logger.error(
                    f"CRITICAL: Failed to remove letter '{letter}' from {player}'s rack {self.player_racks[player]}. Validation/placement inconsistency.")
        logger.debug(f"Rack after removal: {rack}")

    def refill_rack(self, player: str):
        """
        Refills a player's rack to 7 tiles by drawing from the tile bag, if available.
        """
        needed = 7 - len(self.player_racks[player])
        if needed > 0:
            new_tiles = self.draw_tiles(needed)
            if new_tiles:
                self.player_racks[player].extend(new_tiles)
                logger.debug(
                    f"Refilled {player}'s rack with {new_tiles}. New rack: {self.player_racks[player]}")
            else:
                logger.debug(
                    f"Attempted to refill {player}'s rack, but tile bag is empty.")

        if not self.tile_bag:
            logger.info("Tile bag is now empty.")

    def switch_turn(self):
        """Switches the `current_player` attribute between 'human' and 'ai'."""
        self.current_player = "ai" if self.current_player == "human" else "human"
        logger.info(f"--- Turn switched to: {self.current_player} ---")

    def get_opponent(self, player: str) -> str:
        """
        Returns the opponent of the given player ('human' or 'ai').
        """
        return "ai" if player == "human" else "human"

    def get_state(self, hide_ai_rack=True) -> Dict[str, Any]:
        """
        Returns a dictionary representing the current state of the game, suitable for API responses.
        Optionally hides the AI's rack.
        """
        simple_board = [[(cell[0] if cell else None)
                         for cell in row] for row in self.board]

        state: Dict[str, Any] = {
            "board": simple_board,
            "scores": self.scores,
            "current_player": self.current_player,
            "racks": {
                "human": self.player_racks["human"],
            },
            "game_over": self.game_over,
            "first_move": self.first_move,
            "tiles_in_bag": len(self.tile_bag)
        }
        if not hide_ai_rack:
            state["racks"]["ai"] = self.player_racks["ai"]
        return state


# --------------------------
# AI Player Logic
# --------------------------

class AIPlayer:
    """
    Contains static methods for generating and selecting the AI's move using a greedy approach.
    """

    @staticmethod
    def get_best_move(board: Board) -> Optional[Dict[str, Any]]:
        """
        Determines the highest scoring valid move for the AI based on the current board state.
        Returns the move details or None if the AI should pass.
        """
        start_time = time.time()
        ai_player_key = "ai"
        logger.info(
            f"AI starting move generation. Rack: {board.player_racks[ai_player_key]}")

        potential_moves = AIPlayer.generate_all_potential_placements(
            board, ai_player_key)
        logger.info(
            f"AI generated {len(potential_moves)} potential placements.")

        best_move_info = None
        best_score = -1

        validated_moves = []

        for move_candidate in potential_moves:
            word = move_candidate["word"]
            row = move_candidate["row"]
            col = move_candidate["col"]
            direction = move_candidate["direction"]

            is_valid, msg, placed_info = board.validate_move(
                word, row, col, direction, ai_player_key)

            if is_valid:
                try:
                    formed_word_for_score = AIPlayer._get_formed_word(
                        board, word, row, col, direction, placed_info)

                    start_r_score, start_c_score = row, col
                    if direction == "horizontal":
                        while start_c_score > 0 and board.get_letter_at(row, start_c_score - 1):
                            start_c_score -= 1
                    else:
                        while start_r_score > 0 and board.get_letter_at(start_r_score - 1, col):
                            start_r_score -= 1

                    score, words_formed = board.calculate_move_score(
                        formed_word_for_score, start_r_score, start_c_score, direction, placed_info
                    )

                    validated_moves.append({
                        "word": word,
                        "row": row,
                        "col": col,
                        "direction": direction,
                        "score": score,
                        "formed_word": formed_word_for_score,
                        "words_formed": words_formed
                    })
                    logger.debug(
                        f"  Valid AI move found: {word} at ({row},{col}) {direction}, Score: {score}, Formed: {formed_word_for_score}, Crosswords: {words_formed}")
                except Exception as e:
                    logger.error(
                        f"Error scoring valid AI move candidate {move_candidate}: {e}", exc_info=True)

        if validated_moves:
            validated_moves.sort(key=lambda x: x["score"], reverse=True)
            best_move_info = validated_moves[0]
            best_score = best_move_info["score"]
            logger.info(
                f"AI chose best move: {best_move_info['formed_word']} (proposed: {best_move_info['word']}) at ({best_move_info['row']},{best_move_info['col']}) {best_move_info['direction']} with score {best_score}. Other words: {best_move_info['words_formed']}")
        else:
            logger.info(
                "AI found no valid moves after checking generated candidates.")
            best_move_info = None

        end_time = time.time()
        logger.info(
            f"AI move generation and validation took {end_time - start_time:.3f} seconds.")

        if best_move_info:
            return {
                "word": best_move_info["word"],
                "row": best_move_info["row"],
                "col": best_move_info["col"],
                "direction": best_move_info["direction"]
            }
        else:
            return None

    @staticmethod
    def _get_formed_word(board: Board, proposed_word: str, row: int, col: int, direction: str, placed_info: List[Dict[str, Any]]) -> str:
        """
        Helper function to reconstruct the full word actually formed on the board by a placement,
        including adjacent existing letters.
        """
        parts = []
        placed_map = {p['pos']: p['letter'] for p in placed_info}

        if direction == "horizontal":
            start_c = col
            while start_c > 0 and board.get_letter_at(row, start_c - 1):
                start_c -= 1

            max_c_placed = max(
                (p['pos'][1] for p in placed_info), default=col-1) if placed_info else col-1
            end_c = max(col + len(proposed_word) - 1, max_c_placed)
            while end_c < board.size - 1 and (board.get_letter_at(row, end_c + 1) or (row, end_c + 1) in placed_map):
                end_c += 1

            for c_idx in range(start_c, end_c + 1):
                placed = placed_map.get((row, c_idx))
                existing = board.get_letter_at(row, c_idx)
                if placed:
                    parts.append(placed)
                elif existing:
                    parts.append(existing)

        else:
            start_r = row
            while start_r > 0 and board.get_letter_at(start_r - 1, col):
                start_r -= 1

            max_r_placed = max(
                (p['pos'][0] for p in placed_info), default=row-1) if placed_info else row-1
            end_r = max(row + len(proposed_word) - 1, max_r_placed)
            while end_r < board.size - 1 and (board.get_letter_at(end_r + 1, col) or (end_r + 1, col) in placed_map):
                end_r += 1

            for r_idx in range(start_r, end_r + 1):
                placed = placed_map.get((r_idx, col))
                existing = board.get_letter_at(r_idx, col)
                if placed:
                    parts.append(placed)
                elif existing:
                    parts.append(existing)

        formed_word = "".join(parts)

        check_ok = True
        temp_formed_list = list(formed_word)
        for info in placed_info:
            try:
                temp_formed_list.remove(info['letter'])
            except ValueError:
                check_ok = False
                break
        if not check_ok:
            logger.warning(
                f"Formed word reconstruction mismatch? Proposed: {proposed_word}, Placed: {placed_info}, Formed: {formed_word}")

        return formed_word

    @staticmethod
    def generate_all_potential_placements(board: Board, player: str) -> List[Dict[str, Any]]:
        """
        Generates a list of potential word placement candidates for the AI,
        anchored to existing tiles or the center. Uses rack permutations.
        Validation occurs separately.
        """
        potential_placements = []
        rack = board.player_racks[player]
        rack_letters = "".join(rack)
        num_blanks = rack.count(' ')
        anchor_squares = find_anchor_squares(board)

        logger.debug(
            f"AI generating potential placements. Rack: {rack}, Anchors: {len(anchor_squares)}")

        candidate_words = set()
        max_len = len(rack)
        min_len = 2

        for length in range(min_len, max_len + 1):
            for p in permutations(rack_letters, length):
                word = "".join(p).upper()
                if length > num_blanks or any(c != ' ' for c in word):
                    candidate_words.add(word)

        logger.debug(
            f"Generated {len(candidate_words)} unique candidate strings from rack permutations.")

        checked_placements = set()

        for word in candidate_words:
            word_len = len(word)
            for r_anchor, c_anchor in anchor_squares:
                for i in range(word_len):
                    start_col = c_anchor - i
                    start_row = r_anchor
                    if start_col >= 0 and start_col + word_len <= board.size:
                        placement_key = (
                            word, start_row, start_col, "horizontal")
                        if placement_key not in checked_placements:
                            potential_placements.append({
                                "word": word, "row": start_row, "col": start_col, "direction": "horizontal"
                            })
                            checked_placements.add(placement_key)

                for i in range(word_len):
                    start_row = r_anchor - i
                    start_col = c_anchor
                    if start_row >= 0 and start_row + word_len <= board.size:
                        placement_key = (
                            word, start_row, start_col, "vertical")
                        if placement_key not in checked_placements:
                            potential_placements.append({
                                "word": word, "row": start_row, "col": start_col, "direction": "vertical"
                            })
                            checked_placements.add(placement_key)

        logger.debug(
            f"Total potential placement candidates generated: {len(potential_placements)}")
        return potential_placements


def find_anchor_squares(board: Board) -> List[Tuple[int, int]]:
    """
    Identifies potential "anchor" squares (empty squares adjacent to existing tiles)
    for placing new words, or the center square on the first move.
    """
    center_square = (board.size // 2, board.size // 2)
    if board.first_move:
        return [center_square]

    anchors = set()
    for r in range(board.size):
        for c in range(board.size):
            if board.get_letter_at(r, c) is None:
                is_anchor = False
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < board.size and 0 <= nc < board.size and board.get_letter_at(nr, nc):
                        is_anchor = True
                        break
                if is_anchor:
                    anchors.add((r, c))

    return list(anchors) if anchors else [center_square]

# --------------------------
# API Request/Response Models
# --------------------------


class MoveRequest(BaseModel):
    """Defines the expected structure for a player's move request."""
    word: str
    row: int
    col: int
    direction: str


class GameStateResponse(BaseModel):
    """Defines the structure of the game state response sent to the client."""
    board: List[List[Optional[str]]]
    scores: Dict[str, int]
    current_player: str
    player_rack: List[str]
    game_over: bool
    message: Optional[str] = None
    first_move: bool
    tiles_in_bag: int

# --------------------------
# Global Game Instance
# --------------------------


game_board = Board()


# --------------------------
# API Endpoints
# --------------------------

@app.get("/api/game/start", response_model=GameStateResponse, tags=["Game Flow"])
async def start_game():
    """
    Initializes and starts a new Scrabble game, resetting the global game state.
    """
    global game_board
    game_board = Board()
    logger.info("--- New Game Started ---")
    state = game_board.get_state()
    return GameStateResponse(
        board=state["board"],
        scores=state["scores"],
        current_player=state["current_player"],
        player_rack=state["racks"]["human"],
        game_over=state["game_over"],
        message="New game started. Your turn.",
        first_move=state["first_move"],
        tiles_in_bag=state["tiles_in_bag"]
    )


@app.post("/api/game/move", response_model=GameStateResponse, tags=["Game Actions"])
async def player_move(move: MoveRequest):
    """
    Processes a move submitted by the human player, validates it, updates state,
    and triggers the AI's turn if applicable. Returns the updated game state.
    """
    global game_board
    start_turn_time = time.time()
    player = "human"

    if game_board.game_over:
        logger.warning("Move attempted after game over.")
        raise HTTPException(status_code=400, detail="Game is over.")
    if game_board.current_player != player:
        logger.warning(
            f"Move attempt by {player} but it's {game_board.current_player}'s turn.")
        raise HTTPException(status_code=400, detail="Not your turn.")

    success, human_message, human_score = game_board.place_word(
        move.word, move.row, move.col, move.direction, player
    )

    if not success:
        state = game_board.get_state()
        return GameStateResponse(
            board=state["board"], scores=state["scores"], current_player=state["current_player"],
            player_rack=state["racks"]["human"], game_over=state["game_over"],
            message=f"Invalid Move: {human_message}",
            first_move=state["first_move"], tiles_in_bag=state["tiles_in_bag"]
        )

    ai_message = ""
    if not game_board.game_over:
        if game_board.current_player == "ai":
            logger.info("AI turn starting...")
            ai_move = AIPlayer.get_best_move(game_board)

            if ai_move:
                ai_success, message_ai, ai_score = game_board.place_word(
                    ai_move["word"], ai_move["row"], ai_move["col"], ai_move["direction"], "ai"
                )
                if ai_success:
                    ai_message = message_ai
                else:
                    logger.error(
                        f"AI failed to place supposedly valid move {ai_move}: {message_ai}. AI Passing.")
                    ai_pass_success, ai_pass_message = game_board.pass_turn(
                        "ai")
                    ai_message = f"AI Error ({message_ai}). AI passes. ({ai_pass_message})"

            else:
                logger.info("AI found no valid moves or decided to pass.")
                ai_pass_success, ai_pass_message = game_board.pass_turn("ai")
                ai_message = f"AI passes. ({ai_pass_message})"
        else:
            logger.error(
                f"State inconsistency: After human move, current player is {game_board.current_player}, not AI.")
            ai_message = "[Internal Error: Turn order inconsistent]"

    final_message = f"Your move: {human_message}"
    if ai_message:
        final_message += f" || AI move: {ai_message}"

    if game_board.game_over:
        final_message += " || GAME OVER."
        final_message += f" Final Score -> You: {game_board.scores.get('human', 0)}, AI: {game_board.scores.get('ai', 0)}"

    final_state = game_board.get_state()
    end_turn_time = time.time()
    logger.info(
        f"Full turn processing time (human move + AI response): {end_turn_time - start_turn_time:.3f} seconds.")

    return GameStateResponse(
        board=final_state["board"],
        scores=final_state["scores"],
        current_player=final_state["current_player"],
        player_rack=final_state["racks"]["human"],
        game_over=final_state["game_over"],
        message=final_message,
        first_move=final_state["first_move"],
        tiles_in_bag=final_state["tiles_in_bag"]
    )


@app.post("/api/game/pass", response_model=GameStateResponse, tags=["Game Actions"])
async def player_pass():
    """
    Processes a request from the human player to pass their turn, updates state,
    and triggers the AI's turn if applicable. Returns the updated game state.
    """
    global game_board
    start_turn_time = time.time()
    player = "human"

    if game_board.game_over:
        logger.warning("Pass attempted after game over.")
        raise HTTPException(status_code=400, detail="Game is over.")
    if game_board.current_player != player:
        logger.warning(
            f"Pass attempt by {player} but it's {game_board.current_player}'s turn.")
        raise HTTPException(status_code=400, detail="Not your turn.")

    success, pass_message = game_board.pass_turn(player)
    if not success:
        logger.error("Error processing human pass turn internally.")
        raise HTTPException(status_code=500, detail="Error processing pass.")

    human_message = pass_message

    ai_message = ""
    if not game_board.game_over:
        if game_board.current_player == "ai":
            logger.info("AI turn starting after human pass...")
            ai_move = AIPlayer.get_best_move(game_board)
            if ai_move:
                ai_success, message_ai, ai_score = game_board.place_word(
                    ai_move["word"], ai_move["row"], ai_move["col"], ai_move["direction"], "ai"
                )
                if ai_success:
                    ai_message = message_ai
                else:
                    logger.error(
                        f"AI failed to place supposedly valid move {ai_move} after human pass: {message_ai}. AI Passing.")
                    ai_pass_success, ai_pass_message = game_board.pass_turn(
                        "ai")
                    ai_message = f"AI Error ({message_ai}). AI passes. ({ai_pass_message})"
            else:
                logger.info("AI found no valid moves or decided to pass.")
                ai_pass_success, ai_pass_message = game_board.pass_turn("ai")
                ai_message = f"AI passes. ({ai_pass_message})"
        else:
            logger.error(
                f"State inconsistency: After human pass, current player is {game_board.current_player}, not AI.")
            ai_message = "[Internal Error: Turn order inconsistent]"

    final_message = f"You passed."
    if ai_message:
        final_message += f" || AI move: {ai_message}"

    if game_board.game_over:
        final_message += " || GAME OVER."
        final_message += f" Final Score -> You: {game_board.scores.get('human', 0)}, AI: {game_board.scores.get('ai', 0)}"

    final_state = game_board.get_state()
    end_turn_time = time.time()
    logger.info(
        f"Full turn processing time (human pass + AI response): {end_turn_time - start_turn_time:.3f} seconds.")

    return GameStateResponse(
        board=final_state["board"],
        scores=final_state["scores"],
        current_player=final_state["current_player"],
        player_rack=final_state["racks"]["human"],
        game_over=final_state["game_over"],
        message=final_message,
        first_move=final_state["first_move"],
        tiles_in_bag=final_state["tiles_in_bag"]
    )


@app.get("/api/game/state", response_model=GameStateResponse, tags=["Game Info"])
async def get_current_game_state():
    """
    Retrieves the current state of the game without making any changes.
    """
    global game_board
    state = game_board.get_state()
    return GameStateResponse(
        board=state["board"],
        scores=state["scores"],
        current_player=state["current_player"],
        player_rack=state["racks"]["human"],
        game_over=state["game_over"],
        message="Current game state retrieved.",
        first_move=state["first_move"],
        tiles_in_bag=state["tiles_in_bag"]
    )


# --------------------------
# Main Execution Guard
# --------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Scrabble backend server...")
    if not VALID_WORDS or len(VALID_WORDS) < 50:
        logger.critical(
            f"Word dictionary seems invalid or too small ({len(VALID_WORDS)} words). "
            "Please ensure 'scrabble_words.txt' exists, is readable, and contains valid words. Exiting."
        )
        exit(1)

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
