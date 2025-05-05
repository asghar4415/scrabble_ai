from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Tuple, Set, Any
import random
from copy import deepcopy
import logging
import time
import json
import os
from itertools import permutations
import math  # Needed for math.inf in minimax

# --------------------------
# Initialization and Configuration
# --------------------------

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Advanced Scrabble AI Game Backend")

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
    'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1, 'F': 4, 'G': 2, 'H': 4,
    'I': 1, 'J': 8, 'K': 5, 'L': 1, 'M': 3, 'N': 1, 'O': 1, 'P': 3,
    'Q': 10, 'R': 1, 'S': 1, 'T': 1, 'U': 1, 'V': 4, 'W': 4, 'X': 8,
    'Y': 4, 'Z': 10, ' ': 0,
    # Power tile scores (if they reach the board, normally effects trigger)
    'D': 0,  # Placeholder for Double Turn Tile 'D*'
}

# --- Power Tile Definitions ---
# Represents a Double Turn tile in the bag/rack
POWER_TILE_DOUBLE_TURN_MARKER = 'D*'
POWER_TILE_TYPES = {
    POWER_TILE_DOUBLE_TURN_MARKER: {
        "effect": "double_turn", "display": "D"}  # Display 'D' on tile
}

# --- Dynamic Objective Definitions ---
OBJECTIVE_TYPES = [
    {"id": "score_gt_30", "desc": "Score 30+ points in a single turn", "bonus": 20},
    {"id": "use_q_z_x_j", "desc": "Play a word using Q, Z, X, or J", "bonus": 15},
    {"id": "form_7_letter",
        "desc": "Form a 7-letter word (Bingo already gives 50)", "bonus": 25},
    {"id": "use_corner",
        "desc": "Play a tile on a corner square (0,0/0,14/14,0/14,14)", "bonus": 10},
]


def initialize_dictionary() -> Set[str]:
    words = set()
    possible_paths = ['scrabble_words.txt', os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'scrabble_words.txt'), './scrabble_words.txt']
    dict_path_found = next(
        (path for path in possible_paths if os.path.exists(path)), None)

    minimal_word_set = {"QI", "ZA", "CAT", "DOG", "JO", "AX", "EX",
                        "OX", "XI", "XU", "WORD", "PLAY", "GAME", "POWER", "TURN"}

    if not dict_path_found:
        logger.warning("scrabble_words.txt not found. Using minimal word set.")
        return minimal_word_set

    try:
        with open(dict_path_found, 'r', encoding='utf-8') as f:
            loaded_words = {line.strip().upper() for line in f if len(
                line.strip()) >= 2 and line.strip().isalpha()}
        if not loaded_words:
            logger.warning(
                f"{dict_path_found} empty or invalid, using minimal set.")
            words = minimal_word_set
        else:
            words = loaded_words
            logger.info(f"Loaded {len(words)} words from {dict_path_found}")
        return words
    except Exception as e:
        logger.error(
            f"Error reading {dict_path_found}: {e}. Using minimal word set.")
        return minimal_word_set


VALID_WORDS = initialize_dictionary()


def is_valid_word(word: str) -> bool:
    if not word or len(word) < 2 or not word.isalpha():
        return False
    return word.upper() in VALID_WORDS

# --------------------------
# Game Board Class
# --------------------------


class Board:
    def __init__(self):
        self.size = 15
        self.board: List[List[Optional[Tuple[str, bool]]]] = [
            [None for _ in range(self.size)] for _ in range(self.size)]
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
        self.player_objectives = {
            "human": self.assign_objective(),
            "ai": self.assign_objective()
        }
        self.power_tile_effect_active = {
            "double_turn": False}  # Track active effects

    def get_letter_at(self, r: int, c: int) -> Optional[str]:
        if 0 <= r < self.size and 0 <= c < self.size and self.board[r][c]:
            # Return the display letter, not the internal marker like D*
            tile_marker = self.board[r][c][0]
            return POWER_TILE_TYPES.get(tile_marker, {}).get("display", tile_marker)
        return None

    def get_raw_tile_marker_at(self, r: int, c: int) -> Optional[str]:
        if 0 <= r < self.size and 0 <= c < self.size and self.board[r][c]:
            return self.board[r][c][0]  # Returns 'A', ' ', 'D*', etc.
        return None

    def initialize_tile_bag(self) -> List[str]:
        distribution = {
            'A': 9, 'B': 2, 'C': 2, 'D': 4, 'E': 12, 'F': 2, 'G': 3, 'H': 2,
            'I': 9, 'J': 1, 'K': 1, 'L': 4, 'M': 2, 'N': 6, 'O': 8, 'P': 2,
            'Q': 1, 'R': 6, 'S': 4, 'T': 6, 'U': 4, 'V': 2, 'W': 2, 'X': 1,
            'Y': 2, 'Z': 1, ' ': 2
        }
        # Add Power Tiles (Example: 1 Double Turn tile)
        power_tiles_to_add = {POWER_TILE_DOUBLE_TURN_MARKER: 1}

        tile_bag = []
        for letter, count in distribution.items():
            tile_bag.extend([letter] * count)
        for pt_marker, count in power_tiles_to_add.items():
            tile_bag.extend([pt_marker] * count)

        random.shuffle(tile_bag)
        logger.info(
            f"Initialized tile bag with {len(tile_bag)} tiles (incl. power tiles).")
        return tile_bag

    def draw_tiles(self, count: int) -> List[str]:
        drawn = []
        for _ in range(count):
            if not self.tile_bag:
                break
            drawn.append(self.tile_bag.pop())
        return drawn

    def initialize_premium_squares(self) -> Dict[Tuple[int, int], str]:
        premiums = {}
        size = self.size
        center = size // 2
        tw_coords = [(0, 0), (0, 7), (7, 0)]
        dw_coords = [(r, r) for r in range(1, 5)] + [(center, center)
                                                     ] + [(r, size - 1 - r) for r in range(1, 5)]
        tl_coords = [(1, 5), (1, 9), (5, 1), (5, 5), (5, 9), (5, 13)]
        dl_coords = [(0, 3), (0, 11), (2, 6), (2, 8), (3, 0), (3, 7), (3, 14),
                     (6, 2), (6, 6), (6, 8), (6, 12), (7, 3), (7, 11)]
        all_coords = set((r, c) for r in range(size) for c in range(size))
        def mirror(r, c): return [(r, c), (r, size-1-c),
                                  (size-1-r, c), (size-1-r, size-1-c)]
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

    def assign_objective(self) -> Dict[str, Any]:
        objective = random.choice(OBJECTIVE_TYPES)
        return {"id": objective["id"], "desc": objective["desc"], "bonus": objective["bonus"], "completed": False}

    def calculate_move_score(self, word: str, row: int, col: int, direction: str,
                             placed_tiles_info: List[Dict[str, Any]]) -> Tuple[int, List[str]]:
        total_score = 0
        words_formed = []
        main_word_score = 0
        main_word_multiplier = 1
        placed_positions = {info['pos'] for info in placed_tiles_info}

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

                # Power tiles usually don't have score value, their effect triggers elsewhere
                raw_tile = next(
                    (info['tile_marker'] for info in placed_tiles_info if info['pos'] == current_pos), None)
                if raw_tile in POWER_TILE_TYPES:
                    letter_value = 0

                if square_type == 'DL':
                    letter_multiplier = 2
                elif square_type == 'TL':
                    letter_multiplier = 3
                elif square_type == 'DW':
                    main_word_multiplier *= 2
                elif square_type == 'TW':
                    main_word_multiplier *= 3

            main_word_score += letter_value * letter_multiplier

        main_word_score *= main_word_multiplier
        total_score += main_word_score
        words_formed.append(word)

        cross_direction = "vertical" if direction == "horizontal" else "horizontal"
        for info in placed_tiles_info:
            r_placed, c_placed = info['pos']
            placed_letter = info['letter']  # Display letter
            is_blank_placed = info['is_blank']

            cross_word_list = [placed_letter]
            cross_word_start_pos = (r_placed, c_placed)

            # Extend backwards
            cr, cc = r_placed, c_placed
            while True:
                nr, nc = (
                    cr - 1, cc) if cross_direction == "vertical" else (cr, cc - 1)
                if not (0 <= nr < self.size and 0 <= nc < self.size):
                    break
                letter_to_add = self.get_letter_at(
                    nr, nc)  # Use display letter getter
                if letter_to_add:
                    cross_word_list.insert(0, letter_to_add)
                    cross_word_start_pos = (nr, nc)
                    cr, cc = nr, nc
                else:
                    break

            # Extend forwards
            cr, cc = r_placed, c_placed
            while True:
                nr, nc = (
                    cr + 1, cc) if cross_direction == "vertical" else (cr, cc + 1)
                if not (0 <= nr < self.size and 0 <= nc < self.size):
                    break
                letter_to_add = self.get_letter_at(
                    nr, nc)  # Use display letter getter
                if letter_to_add:
                    cross_word_list.append(letter_to_add)
                    cr, cc = nr, nc
                else:
                    break

            cross_word = "".join(cross_word_list)

            if len(cross_word) >= 2 and is_valid_word(cross_word):
                if cross_word not in words_formed:
                    words_formed.append(cross_word)
                    cross_score = 0
                    cross_word_multiplier = 1

                    for i, letter in enumerate(cross_word.upper()):
                        r_cross = cross_word_start_pos[0] + \
                            (i if cross_direction == "vertical" else 0)
                        c_cross = cross_word_start_pos[1] + \
                            (i if cross_direction == "horizontal" else 0)
                        current_pos = (r_cross, c_cross)
                        square_type = self.premium_squares.get(current_pos)

                        letter_value = LETTER_SCORES.get(letter, 0)
                        is_newly_placed_blank_here = False
                        is_newly_placed_power_tile_here = False

                        # Check if the tile at this cross position was placed this turn
                        placed_info_here = next(
                            (p_info for p_info in placed_tiles_info if p_info['pos'] == current_pos), None)
                        if placed_info_here:
                            is_newly_placed_blank_here = placed_info_here['is_blank']
                            is_newly_placed_power_tile_here = placed_info_here[
                                'tile_marker'] in POWER_TILE_TYPES

                        if is_newly_placed_blank_here or is_newly_placed_power_tile_here:
                            letter_value = 0

                        letter_multiplier = 1
                        # Apply premiums only if the anchor tile (the one from the main word) is on the premium sq
                        if current_pos == (r_placed, c_placed):
                            if square_type == 'DL':
                                letter_multiplier = 2
                            elif square_type == 'TL':
                                letter_multiplier = 3
                            if square_type == 'DW':
                                cross_word_multiplier *= 2
                            elif square_type == 'TW':
                                cross_word_multiplier *= 3

                        cross_score += letter_value * letter_multiplier

                    cross_score *= cross_word_multiplier
                    total_score += cross_score

            elif len(cross_word) >= 2:
                logger.warning(
                    f"Invalid crossword '{cross_word}' detected during scoring at {info['pos']}.")
                # In strict rules, this would invalidate the entire move. Here we just log it.

        if len(placed_positions) == 7:
            total_score += 50

        unique_words_formed = sorted(list(set(words_formed)))
        return total_score, unique_words_formed

    def validate_move(self, word_proposal: str, row: int, col: int, direction: str, player: str) -> Tuple[bool, str, List[Dict[str, Any]]]:
        rack = self.player_racks[player]
        temp_board = deepcopy(self.board)  # Use raw board state for validation
        # Stores {'tile_marker': 'A' or 'D*', 'letter': 'A' or 'D', 'pos': (r,c), 'is_blank': bool}
        placed_tiles_info = []
        letters_needed_from_rack = []  # Stores 'A', ' ', 'D*', etc.
        connected_to_existing = self.first_move
        center_square = (self.size // 2, self.size // 2)
        touches_center = False
        num_new_tiles = 0

        # Use display letter for checks if word_proposal comes from frontend/human
        # AI might propose with internal markers, handle that possibility if needed
        word_proposal_upper = word_proposal.upper()

        # Allow digits for power tiles if displayed that way
        if not word_proposal or not all(c.isalnum() or c == ' ' for c in word_proposal_upper):
            return False, "Word proposal invalid.", []

        word_len = len(word_proposal_upper)

        # Basic boundary checks
        if direction == "horizontal":
            if col < 0 or col + word_len > self.size or row < 0 or row >= self.size:
                return False, "Out of bounds.", []
        else:  # Vertical
            if row < 0 or row + word_len > self.size or col < 0 or col >= self.size:
                return False, "Out of bounds.", []

        current_word_parts = []  # Stores display letters for formed word check

        for i, proposed_char in enumerate(word_proposal_upper):
            r = row + (i if direction == "vertical" else 0)
            c = col + (i if direction == "horizontal" else 0)
            pos = (r, c)
            existing_tile_marker = self.get_raw_tile_marker_at(
                r, c)  # Check raw marker on board

            if existing_tile_marker:
                existing_display_letter = POWER_TILE_TYPES.get(
                    existing_tile_marker, {}).get("display", existing_tile_marker)
                if existing_display_letter != proposed_char:
                    return False, f"Conflict at ({r},{c}): Board has '{existing_display_letter}', proposed '{proposed_char}'.", []
                current_word_parts.append(existing_display_letter)
                connected_to_existing = True
            else:
                # This is a newly placed tile
                num_new_tiles += 1
                current_word_parts.append(proposed_char)
                # We need to figure out which TILE MARKER from the rack ('A', ' ', 'D*') corresponds
                # to this proposed_char ('A', 'A', 'D'). This happens below.
                letters_needed_from_rack.append(
                    proposed_char)  # Use proposed char for now
                # Placeholder info, 'tile_marker' and 'is_blank' filled later
                placed_tiles_info.append(
                    {'letter': proposed_char, 'pos': pos, 'is_blank': False, 'tile_marker': '?'})

                # Check adjacency for connection
                if not connected_to_existing:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.size and 0 <= nc < self.size and self.get_raw_tile_marker_at(nr, nc):
                            connected_to_existing = True
                            break

            if self.first_move and pos == center_square:
                touches_center = True
            # Update temp_board with display letter for crossword check logic later
            temp_board[r][c] = (proposed_char, False)

        if num_new_tiles == 0:
            return False, "Must place at least one new tile.", []

        # --- Rack Validation: Match proposed letters to rack tiles ('A', ' ', 'D*') ---
        rack_copy = list(rack)
        final_placed_tiles_info = []
        possible = True
        # Copy to track assignments
        unassigned_placed_info = list(placed_tiles_info)

        # Iterate through the tiles needed based on the proposed word structure
        for needed_display_char in letters_needed_from_rack:
            found_match = False
            # 1. Try exact letter match
            if needed_display_char in rack_copy and needed_display_char.isalpha():
                rack_copy.remove(needed_display_char)
                # Find corresponding entry in unassigned_placed_info and update it
                for info in unassigned_placed_info:
                    if info['letter'] == needed_display_char:
                        info['tile_marker'] = needed_display_char
                        info['is_blank'] = False
                        final_placed_tiles_info.append(info)
                        unassigned_placed_info.remove(info)
                        found_match = True
                        break
            # 2. Try matching to a Power Tile display letter
            elif not found_match:
                for pt_marker, pt_data in POWER_TILE_TYPES.items():
                    if pt_data.get("display") == needed_display_char and pt_marker in rack_copy:
                        rack_copy.remove(pt_marker)
                        for info in unassigned_placed_info:
                            if info['letter'] == needed_display_char:
                                info['tile_marker'] = pt_marker
                                # Power tile is not a blank
                                info['is_blank'] = False
                                final_placed_tiles_info.append(info)
                                unassigned_placed_info.remove(info)
                                found_match = True
                                break
                    if found_match:
                        break
            # 3. Try using a blank tile
            if not found_match and ' ' in rack_copy and needed_display_char.isalpha():
                rack_copy.remove(' ')
                for info in unassigned_placed_info:
                    if info['letter'] == needed_display_char:
                        # The marker used was a blank
                        info['tile_marker'] = ' '
                        info['is_blank'] = True
                        final_placed_tiles_info.append(info)
                        unassigned_placed_info.remove(info)
                        found_match = True
                        break

            if not found_match:
                possible = False
                break  # Cannot form the word with available tiles

        if not possible:
            return False, f"Letter '{needed_display_char}' not available in rack {rack} (incl. blanks/power tiles).", []

        if len(final_placed_tiles_info) != len(placed_tiles_info):
            logger.error(
                f"Internal rack validation mismatch. Info: {placed_tiles_info} -> Final: {final_placed_tiles_info}")
            return False, "Internal error during rack validation.", []

        placed_tiles_info = final_placed_tiles_info  # Use the fully resolved info now

        # --- Game Rule Checks ---
        if self.first_move and not touches_center:
            return False, "First move must touch the center square.", []
        if not self.first_move and not connected_to_existing:
            return False, "Word must connect to existing tiles.", []

        # --- Word Validity Checks ---
        formed_word = "".join(current_word_parts)
        if not is_valid_word(formed_word):
            return False, f"Main word '{formed_word}' is not valid.", []

        # Check crosswords (using display letters on temp_board)
        cross_direction = "vertical" if direction == "horizontal" else "horizontal"
        for info in placed_tiles_info:
            r_placed, c_placed = info['pos']
            placed_display_letter = info['letter']
            cross_word_list = [placed_display_letter]

            # Extend backwards
            cr, cc = r_placed, c_placed
            while True:
                nr, nc = (
                    cr - 1, cc) if cross_direction == "vertical" else (cr, cc - 1)
                if not (0 <= nr < self.size and 0 <= nc < self.size):
                    break
                # Check the temp board with placed display letters
                tile_info = temp_board[nr][nc]
                if tile_info:
                    cross_word_list.insert(0, tile_info[0])
                    cr, cc = nr, nc
                else:
                    break

            # Extend forwards
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
                return False, f"Creates invalid crossword '{cross_word}' at ({r_placed},{c_placed}).", []

        return True, f"Move '{formed_word}' is valid.", placed_tiles_info

    def place_word(self, word: str, row: int, col: int, direction: str, player: str) -> Tuple[bool, str, Optional[int]]:
        logger.info(
            f"Attempting move by {player}: word='{word}', pos=({row},{col}), dir={direction}")
        is_valid, message, placed_tiles_info = self.validate_move(
            word, row, col, direction, player)

        if not is_valid:
            logger.warning(f"Validation failed for {player}: {message}")
            return False, message, None

        # Reconstruct the main word using the validated info (incl. existing tiles)
        # Need a robust way to get the full word as formed on the board
        formed_word = self._get_formed_word_from_placement(
            row, col, direction, placed_tiles_info)
        logger.info(f"Validated formed word: '{formed_word}'")

        # Find the true start of the main word for scoring
        start_r_score, start_c_score = self._get_word_start(
            row, col, direction)

        # Calculate score
        score, words_formed = self.calculate_move_score(
            formed_word, start_r_score, start_c_score, direction, placed_tiles_info
        )
        logger.info(
            f"Player {player} formed words: {words_formed} for base score: {score}")

        # --- Check and Apply Objectives ---
        objective_bonus = 0
        objective_msg = ""
        objective = self.player_objectives[player]
        if not objective["completed"]:
            completed_now = self.check_objective_completion(
                objective["id"], score, words_formed, placed_tiles_info)
            if completed_now:
                objective["completed"] = True
                objective_bonus = objective["bonus"]
                score += objective_bonus
                objective_msg = f" Objective '{objective['desc']}' completed (+{objective_bonus} pts)!"
                logger.info(
                    f"{player} completed objective {objective['id']} for {objective_bonus} points.")

        # --- Place Tiles and Trigger Effects ---
        letters_removed_from_rack = []
        power_tile_triggered = None
        for info in placed_tiles_info:
            r_place, c_place = info['pos']
            # Use the actual marker ('A', ' ', 'D*')
            tile_marker = info['tile_marker']
            self.board[r_place][c_place] = (
                tile_marker, True)  # Store marker on board
            letters_removed_from_rack.append(tile_marker)

            # Check for power tile effect
            if tile_marker in POWER_TILE_TYPES:
                effect = POWER_TILE_TYPES[tile_marker].get("effect")
                if effect == "double_turn":
                    self.power_tile_effect_active["double_turn"] = True
                    power_tile_triggered = "Double Turn"
                    logger.info(f"{player} played a Double Turn tile!")
                # Add other power tile effects here

        self.scores[player] += score
        self.remove_from_rack(letters_removed_from_rack, player)
        self.refill_rack(player)

        self.first_move = False
        self.consecutive_passes = 0
        self.last_move_time = time.time()

        # Check game over conditions
        if not self.tile_bag and not self.player_racks[player]:
            self.game_over = True
            self.finalize_scores()
            logger.info(
                f"Game Over: {player} used last tile and bag is empty.")

        # Switch turn unless a power tile prevents it
        if not self.game_over:
            if self.power_tile_effect_active["double_turn"]:
                logger.info(
                    f"Double turn activated for {player}. They play again.")
                # Consume effect
                self.power_tile_effect_active["double_turn"] = False
            else:
                self.switch_turn()

        success_message = f"Played '{formed_word}' for {score} points (Base: {score - objective_bonus}). Words: {', '.join(words_formed)}."
        if objective_msg:
            success_message += objective_msg
        if power_tile_triggered:
            success_message += f" Triggered {power_tile_triggered}!"
        return True, success_message, score

    def _get_formed_word_from_placement(self, row: int, col: int, direction: str, placed_info: List[Dict[str, Any]]) -> str:
        """Helper to reconstruct the full primary word formed, including existing tiles."""
        parts = []
        placed_map = {p['pos']: p['letter']
                      for p in placed_info}  # Use display letter

        if direction == "horizontal":
            start_c = col
            while start_c > 0 and self.get_raw_tile_marker_at(row, start_c - 1):
                start_c -= 1
            end_c = col
            max_c_placed = max((p['pos'][1] for p in placed_info),
                               default=col - 1) if placed_info else col - 1
            # Iterate until end of placed word or end of contiguous letters on board
            temp_c = col
            while temp_c < self.size:
                current_pos = (row, temp_c)
                if current_pos in placed_map or self.get_raw_tile_marker_at(row, temp_c):
                    end_c = temp_c
                    temp_c += 1
                else:
                    break

            for c_idx in range(start_c, end_c + 1):
                placed = placed_map.get((row, c_idx))
                # Use display letter getter
                existing = self.get_letter_at(row, c_idx)
                if placed:
                    parts.append(placed)
                elif existing:
                    parts.append(existing)
                # Gaps should be prevented by validation

        else:  # Vertical
            start_r = row
            while start_r > 0 and self.get_raw_tile_marker_at(start_r - 1, col):
                start_r -= 1
            end_r = row
            max_r_placed = max((p['pos'][0] for p in placed_info),
                               default=row - 1) if placed_info else row - 1
            temp_r = row
            while temp_r < self.size:
                current_pos = (temp_r, col)
                if current_pos in placed_map or self.get_raw_tile_marker_at(temp_r, col):
                    end_r = temp_r
                    temp_r += 1
                else:
                    break

            for r_idx in range(start_r, end_r + 1):
                placed = placed_map.get((r_idx, col))
                existing = self.get_letter_at(r_idx, col)
                if placed:
                    parts.append(placed)
                elif existing:
                    parts.append(existing)

        formed_word = "".join(parts)
        return formed_word

    def _get_word_start(self, r_place: int, c_place: int, direction: str) -> Tuple[int, int]:
        """Finds the actual starting cell (top-most or left-most) of a word passing through (r_place, c_place)."""
        start_r, start_c = r_place, c_place
        if direction == "horizontal":
            while start_c > 0 and self.get_raw_tile_marker_at(start_r, start_c - 1):
                start_c -= 1
        else:  # Vertical
            while start_r > 0 and self.get_raw_tile_marker_at(start_r - 1, start_c):
                start_r -= 1
        return start_r, start_c

    def check_objective_completion(self, objective_id: str, move_score: int, words_formed: List[str], placed_tiles_info: List[Dict[str, Any]]) -> bool:
        if objective_id == "score_gt_30":
            return move_score >= 30
        if objective_id == "use_q_z_x_j":
            return any(c in 'QZXJ' for word in words_formed for c in word)
        if objective_id == "form_7_letter":
            # Check if a BINGO was formed (7 new tiles placed)
            return len(placed_tiles_info) == 7
        if objective_id == "use_corner":
            corners = [(0, 0), (0, 14), (14, 0), (14, 14)]
            return any(info['pos'] in corners for info in placed_tiles_info)
        return False

    def pass_turn(self, player: str) -> Tuple[bool, str]:
        if player != self.current_player:
            return False, "Not your turn."

        self.consecutive_passes += 1
        logger.info(
            f"Player {player} passed. Consecutive passes: {self.consecutive_passes}")

        if self.consecutive_passes >= 6:  # Standard Scrabble end condition
            self.game_over = True
            self.finalize_scores()
            logger.info("Game Over: 6 consecutive passes.")
            return True, "Turn passed. Game Over due to 6 consecutive passes."
        else:
            self.switch_turn()
            return True, "Turn passed."

    def finalize_scores(self):
        logger.info("Game Over - Finalizing Scores.")
        empty_rack_player = None
        total_unplayed_score = 0

        for p, rack in self.player_racks.items():
            player_unplayed_score = sum(LETTER_SCORES.get(POWER_TILE_TYPES.get(tile, {}).get(
                "display", tile), 0) for tile in rack)  # Use display letter for score deduction
            if not rack:
                empty_rack_player = p
            else:
                self.scores[p] -= player_unplayed_score
                total_unplayed_score += player_unplayed_score
                logger.info(
                    f"Player {p} deducts {player_unplayed_score} points for unplayed tiles: {rack}")

        if empty_rack_player and self.get_opponent(empty_rack_player) in self.scores:
            self.scores[empty_rack_player] += total_unplayed_score
            logger.info(
                f"Player {empty_rack_player} receives {total_unplayed_score} points.")
        elif empty_rack_player:
            logger.warning(
                "Score finalization: empty rack, but opponent score not found.")
        else:
            logger.info("Game ended with tiles remaining for both players.")

        logger.info(
            f"Final Scores: Human: {self.scores.get('human', 0)}, AI: {self.scores.get('ai', 0)}")

    def remove_from_rack(self, tile_markers: List[str], player: str):
        """Removes tile markers ('A', ' ', 'D*') from rack."""
        rack = self.player_racks[player]
        logger.debug(f"Removing {tile_markers} from {player}'s rack {rack}")
        for marker in tile_markers:
            try:
                rack.remove(marker)
            except ValueError:
                logger.error(
                    f"CRITICAL: Failed to remove marker '{marker}' from {player}'s rack {rack}")
        logger.debug(f"Rack after removal: {rack}")

    def refill_rack(self, player: str):
        needed = 7 - len(self.player_racks[player])
        if needed > 0:
            new_tiles = self.draw_tiles(needed)
            if new_tiles:
                self.player_racks[player].extend(new_tiles)
                logger.debug(
                    f"Refilled {player}'s rack with {new_tiles}. New rack: {self.player_racks[player]}")
        if not self.tile_bag:
            logger.info("Tile bag is empty.")

    def switch_turn(self):
        self.current_player = "ai" if self.current_player == "human" else "human"
        logger.info(f"--- Turn switched to: {self.current_player} ---")

    def get_opponent(self, player: str) -> str:
        return "ai" if player == "human" else "human"

    def get_state(self, hide_ai_rack=True) -> Dict[str, Any]:
        """Returns game state, converting tile markers to display letters."""
        # Board state uses display letters
        display_board = [[(POWER_TILE_TYPES.get(cell[0], {}).get(
            "display", cell[0]) if cell else None) for cell in row] for row in self.board]
        # Rack state uses display letters for frontend
        human_rack_display = [POWER_TILE_TYPES.get(marker, {}).get(
            "display", marker) for marker in self.player_racks["human"]]
        ai_rack_display = []
        if not hide_ai_rack:
            ai_rack_display = [POWER_TILE_TYPES.get(marker, {}).get(
                "display", marker) for marker in self.player_racks["ai"]]

        state: Dict[str, Any] = {
            "board": display_board,
            "scores": self.scores,
            "current_player": self.current_player,
            "racks": {  # API expects racks to have display letters
                "human": human_rack_display,
            },
            "game_over": self.game_over,
            "first_move": self.first_move,
            "tiles_in_bag": len(self.tile_bag),
            # Send human objective status
            "human_objective": self.player_objectives["human"],
            # Optionally add AI objective status if needed, but usually secret
        }
        if not hide_ai_rack:
            state["racks"]["ai"] = ai_rack_display
        return state

    # --- AI Helper Methods ---
    def get_valid_moves(self, player: str) -> List[Dict[str, Any]]:
        """Generates all valid moves for the given player."""
        valid_moves = []
        rack = self.player_racks[player]
        anchor_squares = find_anchor_squares(self)

        potential_placements = generate_potential_placements_anchored(
            self, player, anchor_squares)

        for move_candidate in potential_placements:
            # This is the proposed string using rack letters/blanks/power tile markers
            word = move_candidate["word"]
            row, col, direction = move_candidate["row"], move_candidate["col"], move_candidate["direction"]

            # Convert proposed word using markers ('D*') to word using display letters ('D') for validation
            # This step might be complex if blanks are involved matching power tile display chars.
            # Simplified: Assume `validate_move` handles the marker-to-display mapping internally correctly.
            # `validate_move` needs the proposed word string, row, col, direction.
            # Let's assume the generator gives the right `word` string format for `validate_move`.

            is_valid, msg, placed_info = self.validate_move(
                word, row, col, direction, player)

            if is_valid:
                formed_word = self._get_formed_word_from_placement(
                    row, col, direction, placed_info)
                start_r_score, start_c_score = self._get_word_start(
                    row, col, direction)
                score, words_formed = self.calculate_move_score(
                    formed_word, start_r_score, start_c_score, direction, placed_info)

                # Check for objective completion score *if* this move were made (don't change state)
                objective = self.player_objectives[player]
                objective_bonus = 0
                if not objective["completed"]:
                    if self.check_objective_completion(objective["id"], score, words_formed, placed_info):
                        objective_bonus = objective["bonus"]

                move_data = {
                    # The originally proposed word string (using rack markers)
                    "word": word,
                    "row": row, "col": col, "direction": direction,
                    "score": score + objective_bonus,  # Total potential score for heuristic
                    "base_score": score,
                    "objective_bonus": objective_bonus,
                    "words_formed": words_formed,
                    "placed_info": placed_info  # Needed to apply the move
                }
                valid_moves.append(move_data)

        return valid_moves


# --------------------------
# AI Player Logic (Minimax)
# --------------------------
class AIPlayer:
    MAX_DEPTH = 2  # Adjust depth for performance vs strength
    AI_PLAYER = "ai"
    HUMAN_PLAYER = "human"

    @staticmethod
    def get_best_move(board: Board) -> Optional[Dict[str, Any]]:
        start_time = time.time()
        logger.info(
            f"AI ({AIPlayer.AI_PLAYER}) starting Minimax. Rack: {board.player_racks[AIPlayer.AI_PLAYER]}")

        best_move = None
        best_score = -math.inf

        # Get all valid moves for the AI in the current state
        possible_moves = board.get_valid_moves(AIPlayer.AI_PLAYER)

        if not possible_moves:
            logger.info("AI found no valid moves.")
            return None

        logger.info(
            f"AI evaluating {len(possible_moves)} potential moves using Minimax (Depth {AIPlayer.MAX_DEPTH}).")

        # Iterate through valid moves and evaluate using Minimax
        for move in possible_moves:
            # Create a deep copy of the board state to simulate the move
            temp_board = deepcopy(board)

            # Apply the move to the temporary board (simplified apply logic)
            # Note: A full apply_move function would be cleaner here.
            # This simplified version assumes place_word handles everything correctly,
            # but we only need the state *after* the move for the heuristic.
            # We need to manually update board, score, rack, turn for the temp board.

            # 1. Place tiles on temp_board
            letters_removed = []
            for info in move['placed_info']:
                temp_board.board[info['pos'][0]][info['pos'][1]] = (
                    info['tile_marker'], True)
                letters_removed.append(info['tile_marker'])
            # 2. Update score (use base score before objective bonus for state consistency)
            temp_board.scores[AIPlayer.AI_PLAYER] += move['base_score']
            # If objective completed by this move, add bonus
            if move['objective_bonus'] > 0:
                temp_board.scores[AIPlayer.AI_PLAYER] += move['objective_bonus']
                # Mark completed in temp state
                temp_board.player_objectives[AIPlayer.AI_PLAYER]['completed'] = True
            # 3. Update rack
            temp_board.remove_from_rack(letters_removed, AIPlayer.AI_PLAYER)
            # Refill modifies the tile_bag copy too
            temp_board.refill_rack(AIPlayer.AI_PLAYER)
            # 4. Handle turn switch logic (including double turn)
            power_tile_double_turn = any(
                info['tile_marker'] == POWER_TILE_DOUBLE_TURN_MARKER for info in move['placed_info'])
            if not power_tile_double_turn:
                temp_board.current_player = AIPlayer.HUMAN_PLAYER  # Switch turn in temp state
            else:
                temp_board.current_player = AIPlayer.AI_PLAYER  # AI plays again in temp state
            temp_board.first_move = False
            temp_board.consecutive_passes = 0

            # Call minimax on the resulting state for the opponent
            score = AIPlayer.minimax(
                temp_board, AIPlayer.MAX_DEPTH - 1, -math.inf, math.inf, False)  # Opponent minimizes

            logger.debug(
                f"  Move {move['word']} @ ({move['row']},{move['col']},{move['direction']}), Score: {move['score']}, Minimax eval: {score}")

            # Update best move found so far
            if score > best_score:
                best_score = score
                best_move = move  # Store the move dictionary itself

        end_time = time.time()
        logger.info(
            f"AI Minimax evaluation took {end_time - start_time:.3f} seconds.")

        if best_move:
            logger.info(
                f"AI chose move: {best_move['word']} at ({best_move['row']},{best_move['col']}) {best_move['direction']} with heuristic score {best_score} (Move score: {best_move['score']}). Words: {best_move['words_formed']}")
            # Return only the info needed by place_word
            return {
                "word": best_move["word"],
                "row": best_move["row"],
                "col": best_move["col"],
                "direction": best_move["direction"]
            }
        else:
            logger.info("AI found no suitable move after Minimax evaluation.")
            return None

    @staticmethod
    def minimax(board_state: Board, depth: int, alpha: float, beta: float, is_maximizing_player: bool) -> float:
        """Minimax algorithm with Alpha-Beta Pruning."""

        # Base case: depth reached or game over
        if depth == 0 or board_state.game_over:
            return AIPlayer.evaluate_board(board_state)

        if is_maximizing_player:  # AI's turn (Maximize)
            max_eval = -math.inf
            possible_moves = board_state.get_valid_moves(AIPlayer.AI_PLAYER)
            if not possible_moves:
                # No moves, return static eval
                return AIPlayer.evaluate_board(board_state)

            for move in possible_moves:
                child_state = deepcopy(board_state)
                # Simplified application of move for recursion (needs robust apply_move ideally)
                letters_removed = []
                for info in move['placed_info']:
                    child_state.board[info['pos'][0]][info['pos'][1]] = (
                        info['tile_marker'], True)
                    letters_removed.append(info['tile_marker'])
                child_state.scores[AIPlayer.AI_PLAYER] += move['base_score']
                if move['objective_bonus'] > 0:
                    child_state.scores[AIPlayer.AI_PLAYER] += move['objective_bonus']
                    child_state.player_objectives[AIPlayer.AI_PLAYER]['completed'] = True
                child_state.remove_from_rack(
                    letters_removed, AIPlayer.AI_PLAYER)
                child_state.refill_rack(AIPlayer.AI_PLAYER)
                power_tile_double_turn = any(
                    info['tile_marker'] == POWER_TILE_DOUBLE_TURN_MARKER for info in move['placed_info'])
                # If double turn, AI maximizes again
                next_player_is_maximizing = power_tile_double_turn
                if not power_tile_double_turn:
                    child_state.current_player = AIPlayer.HUMAN_PLAYER
                child_state.first_move = False
                child_state.consecutive_passes = 0

                eval_score = AIPlayer.minimax(
                    child_state, depth - 1, alpha, beta, next_player_is_maximizing)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta cutoff
            return max_eval

        else:  # Opponent's turn (Minimize)
            min_eval = math.inf
            possible_moves = board_state.get_valid_moves(AIPlayer.HUMAN_PLAYER)
            if not possible_moves:
                # No moves, return static eval
                return AIPlayer.evaluate_board(board_state)

            for move in possible_moves:
                child_state = deepcopy(board_state)
                # Simplified application of move for recursion
                letters_removed = []
                for info in move['placed_info']:
                    child_state.board[info['pos'][0]][info['pos'][1]] = (
                        info['tile_marker'], True)
                    letters_removed.append(info['tile_marker'])
                child_state.scores[AIPlayer.HUMAN_PLAYER] += move['base_score']
                if move['objective_bonus'] > 0:
                    child_state.scores[AIPlayer.HUMAN_PLAYER] += move['objective_bonus']
                    child_state.player_objectives[AIPlayer.HUMAN_PLAYER]['completed'] = True
                child_state.remove_from_rack(
                    letters_removed, AIPlayer.HUMAN_PLAYER)
                child_state.refill_rack(AIPlayer.HUMAN_PLAYER)
                power_tile_double_turn = any(
                    info['tile_marker'] == POWER_TILE_DOUBLE_TURN_MARKER for info in move['placed_info'])
                # If human double turn, they minimize again. Else AI maximizes.
                next_player_is_maximizing = not power_tile_double_turn
                if not power_tile_double_turn:
                    child_state.current_player = AIPlayer.AI_PLAYER
                child_state.first_move = False
                child_state.consecutive_passes = 0

                eval_score = AIPlayer.minimax(
                    child_state, depth - 1, alpha, beta, next_player_is_maximizing)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha cutoff
            return min_eval

    @staticmethod
    def evaluate_board(board_state: Board) -> float:
        """
        Heuristic function to evaluate the board state for the AI player.
        Higher score means better for AI.
        """
        ai_score = board_state.scores.get(AIPlayer.AI_PLAYER, 0)
        human_score = board_state.scores.get(AIPlayer.HUMAN_PLAYER, 0)

        # Primary factor: Score difference
        score_diff = ai_score - human_score

        # Bonus for completed objective
        objective_bonus = 50 if board_state.player_objectives[AIPlayer.AI_PLAYER]["completed"] else 0
        # Penalty for opponent completed objective
        opponent_objective_penalty = - \
            50 if board_state.player_objectives[AIPlayer.HUMAN_PLAYER]["completed"] else 0

        # Rack leave value (simple version: penalize having too many vowels/consonants or low-scoring tiles)
        # More advanced: Consider letter synergy, high-value tile potential
        rack_value = 0
        ai_rack = board_state.player_racks[AIPlayer.AI_PLAYER]
        vowels = "AEIOU"
        num_vowels = sum(1 for tile in ai_rack if tile in vowels)
        num_consonants = len(ai_rack) - num_vowels
        rack_balance_penalty = abs(
            num_vowels - num_consonants) * 2  # Penalize imbalance
        rack_value -= rack_balance_penalty

        # Board control (very simple: count tiles placed) - Needs improvement
        ai_tiles_on_board = sum(1 for r in range(board_state.size) for c in range(
            board_state.size) if board_state.get_raw_tile_marker_at(r, c) and board_state.board[r][c][1])  # Check placed flag
        # Could add weights for center, premium squares access

        final_heuristic = score_diff + objective_bonus + \
            opponent_objective_penalty + rack_value + (ai_tiles_on_board * 0.5)

        return final_heuristic


# --- Helper functions for Move Generation ---

def find_anchor_squares(board: Board) -> List[Tuple[int, int]]:
    center_square = (board.size // 2, board.size // 2)
    if board.first_move:
        return [center_square]

    anchors = set()
    for r in range(board.size):
        for c in range(board.size):
            if board.get_raw_tile_marker_at(r, c) is None:  # Empty square
                is_anchor = False
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < board.size and 0 <= nc < board.size and board.get_raw_tile_marker_at(nr, nc):
                        is_anchor = True
                        break
                if is_anchor:
                    anchors.add((r, c))

    # Fallback if somehow board cleared
    return list(anchors) if anchors else [center_square]


def generate_potential_placements_anchored(board: Board, player: str, anchor_squares: List[Tuple[int, int]]) -> List[Dict[str, Any]]:
    """Generates potential word placements anchored to existing tiles."""
    potential_placements = []
    rack = board.player_racks[player]
    rack_letters_with_blanks = list(rack)  # Includes ' ', 'D*', etc.

    if not anchor_squares:
        return []

    # Limit permutations for performance
    MAX_PERM_LEN = min(len(rack), 7)  # Max length to try permutations for
    checked_placements = set()

    for length in range(2, MAX_PERM_LEN + 1):
        # Use set to avoid duplicate permutations
        for p in set(permutations(rack_letters_with_blanks, length)):
            # This string contains markers like ' ' and 'D*'
            perm_str = "".join(p)

            for r_anchor, c_anchor in anchor_squares:
                # Try placing horizontally, anchored at (r_anchor, c_anchor)
                # i is the index within perm_str that lands on the anchor
                for i in range(length):
                    start_col = c_anchor - i
                    start_row = r_anchor
                    if start_col < 0 or start_col + length > board.size:
                        continue

                    # Check if the placement is valid *structurally* (overlaps correctly)
                    # and uses the anchor square with one of the perm_str letters.
                    fixed_value_on_board = False
                    uses_anchor = False
                    # Store the proposed markers ('A', ' ', 'D*') for this placement
                    temp_word_parts = []

                    valid_structure = True
                    # Letters available for this specific permutation
                    current_rack_copy = list(p)

                    for k in range(length):
                        r, c = start_row, start_col + k
                        existing_marker = board.get_raw_tile_marker_at(r, c)
                        proposed_marker = p[k]

                        if existing_marker:  # Overlap
                            # If we overlap, the existing tile must match what we *would* have placed
                            # This is tricky with blanks. Simple check: just ensure it's occupied.
                            # The main `validate_move` will handle detailed checks.
                            # Add existing tile to word structure
                            temp_word_parts.append(existing_marker)
                            fixed_value_on_board = True  # We connected to something
                            if (r, c) == (r_anchor, c_anchor):
                                uses_anchor = True
                        else:  # Placing a new tile
                            temp_word_parts.append(proposed_marker)
                            if (r, c) == (r_anchor, c_anchor):
                                uses_anchor = True

                    # Basic check: placement must use the anchor square and connect to *something* (if not first move)
                    if uses_anchor and (fixed_value_on_board or board.first_move):
                        # Use perm_str as key
                        placement_key = (perm_str, start_row,
                                         start_col, "horizontal")
                        if placement_key not in checked_placements:
                            # We pass the perm_str (with markers) to be validated later
                            potential_placements.append(
                                {"word": perm_str, "row": start_row, "col": start_col, "direction": "horizontal"})
                            checked_placements.add(placement_key)

                # Try placing vertically, anchored at (r_anchor, c_anchor)
                for i in range(length):
                    start_row = r_anchor - i
                    start_col = c_anchor
                    if start_row < 0 or start_row + length > board.size:
                        continue

                    fixed_value_on_board = False
                    uses_anchor = False
                    temp_word_parts = []
                    valid_structure = True
                    current_rack_copy = list(p)

                    for k in range(length):
                        r, c = start_row + k, start_col
                        existing_marker = board.get_raw_tile_marker_at(r, c)
                        proposed_marker = p[k]
                        if existing_marker:
                            temp_word_parts.append(existing_marker)
                            fixed_value_on_board = True
                            if (r, c) == (r_anchor, c_anchor):
                                uses_anchor = True
                        else:
                            temp_word_parts.append(proposed_marker)
                            if (r, c) == (r_anchor, c_anchor):
                                uses_anchor = True

                    if uses_anchor and (fixed_value_on_board or board.first_move):
                        placement_key = (perm_str, start_row,
                                         start_col, "vertical")
                        if placement_key not in checked_placements:
                            potential_placements.append(
                                {"word": perm_str, "row": start_row, "col": start_col, "direction": "vertical"})
                            checked_placements.add(placement_key)

    logger.debug(
        f"Generated {len(potential_placements)} potential anchored placements for {player}.")
    return potential_placements


# --------------------------
# API Request/Response Models
# --------------------------
class MoveRequest(BaseModel):
    word: str
    row: int
    col: int
    direction: str


class GameStateResponse(BaseModel):
    board: List[List[Optional[str]]]
    scores: Dict[str, int]
    current_player: str
    player_rack: List[str]  # Expects display letters
    game_over: bool
    message: Optional[str] = None
    first_move: bool
    tiles_in_bag: int
    human_objective: Optional[Dict[str, Any]
                              ] = None  # Include objective status


# --------------------------
# Global Game Instance
# --------------------------
game_board = Board()

# --------------------------
# API Endpoints
# --------------------------


@app.get("/api/game/start", response_model=GameStateResponse, tags=["Game Flow"])
async def start_game():
    global game_board
    game_board = Board()
    logger.info("--- New Advanced Scrabble Game Started ---")
    state = game_board.get_state()
    return GameStateResponse(**state, player_rack=state["racks"]["human"], message="New game started. Your turn.")


@app.post("/api/game/move", response_model=GameStateResponse, tags=["Game Actions"])
async def player_move(move: MoveRequest):
    global game_board
    start_turn_time = time.time()
    player = "human"

    if game_board.game_over:
        raise HTTPException(status_code=400, detail="Game is over.")
    if game_board.current_player != player:
        raise HTTPException(status_code=400, detail="Not your turn.")

    success, human_message, human_score = game_board.place_word(
        move.word, move.row, move.col, move.direction, player
    )

    if not success:
        state = game_board.get_state()
        # Need to map internal state structure to response model
        return GameStateResponse(**state, player_rack=state["racks"]["human"], message=f"Invalid Move: {human_message}")

    # --- AI Turn ---
    ai_message = ""
    if not game_board.game_over and game_board.current_player == "ai":
        logger.info("AI turn starting...")
        ai_move_action = AIPlayer.get_best_move(
            game_board)  # Returns dict or None

        if ai_move_action:
            ai_success, message_ai, ai_score = game_board.place_word(
                ai_move_action["word"], ai_move_action["row"], ai_move_action["col"], ai_move_action["direction"], "ai"
            )
            if ai_success:
                ai_message = message_ai
            else:
                logger.error(
                    f"AI failed to place supposedly valid move {ai_move_action}: {message_ai}. AI Passing.")
                ai_pass_success, ai_pass_message = game_board.pass_turn("ai")
                ai_message = f"AI Error ({message_ai}). AI passes. ({ai_pass_message})"
        else:
            logger.info("AI found no valid moves or decided to pass.")
            ai_pass_success, ai_pass_message = game_board.pass_turn("ai")
            ai_message = f"AI passes. ({ai_pass_message})"

    # --- Prepare Response ---
    final_message = f"Your move: {human_message}"
    if ai_message:
        final_message += f" || AI move: {ai_message}"
    if game_board.game_over and "GAME OVER" not in final_message.upper():
        final_message += " || GAME OVER."
        final_message += f" Final Score -> You: {game_board.scores.get('human', 0)}, AI: {game_board.scores.get('ai', 0)}"

    final_state = game_board.get_state()
    end_turn_time = time.time()
    logger.info(
        f"Full turn processing time: {end_turn_time - start_turn_time:.3f} seconds.")

    return GameStateResponse(**final_state, player_rack=final_state["racks"]["human"], message=final_message)


@app.post("/api/game/pass", response_model=GameStateResponse, tags=["Game Actions"])
async def player_pass():
    global game_board
    start_turn_time = time.time()
    player = "human"

    if game_board.game_over:
        raise HTTPException(status_code=400, detail="Game is over.")
    if game_board.current_player != player:
        raise HTTPException(status_code=400, detail="Not your turn.")

    success, pass_message = game_board.pass_turn(player)
    if not success:
        raise HTTPException(status_code=500, detail="Error processing pass.")

    # --- AI Turn ---
    ai_message = ""
    if not game_board.game_over and game_board.current_player == "ai":
        logger.info("AI turn starting after human pass...")
        ai_move_action = AIPlayer.get_best_move(game_board)
        if ai_move_action:
            ai_success, message_ai, ai_score = game_board.place_word(
                ai_move_action["word"], ai_move_action["row"], ai_move_action["col"], ai_move_action["direction"], "ai"
            )
            if ai_success:
                ai_message = message_ai
            else:
                logger.error(
                    f"AI failed to place supposedly valid move {ai_move_action} after human pass: {message_ai}. AI Passing.")
                ai_pass_success, ai_pass_message = game_board.pass_turn("ai")
                ai_message = f"AI Error ({message_ai}). AI passes. ({ai_pass_message})"
        else:
            logger.info("AI found no valid moves or decided to pass.")
            ai_pass_success, ai_pass_message = game_board.pass_turn("ai")
            ai_message = f"AI passes. ({ai_pass_message})"

    # --- Prepare Response ---
    final_message = f"You passed."
    if ai_message:
        final_message += f" || AI move: {ai_message}"
    if game_board.game_over and "GAME OVER" not in final_message.upper():
        final_message += " || GAME OVER."
        final_message += f" Final Score -> You: {game_board.scores.get('human', 0)}, AI: {game_board.scores.get('ai', 0)}"

    final_state = game_board.get_state()
    end_turn_time = time.time()
    logger.info(
        f"Full turn processing time (pass): {end_turn_time - start_turn_time:.3f} seconds.")

    return GameStateResponse(**final_state, player_rack=final_state["racks"]["human"], message=final_message)


@app.get("/api/game/state", response_model=GameStateResponse, tags=["Game Info"])
async def get_current_game_state():
    global game_board
    state = game_board.get_state()
    return GameStateResponse(**state, player_rack=state["racks"]["human"], message="Current game state retrieved.")

# --------------------------
# Main Execution Guard
# --------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Advanced Scrabble backend server...")
    if not VALID_WORDS or len(VALID_WORDS) < 50:
        logger.critical(
            f"Word dictionary invalid or too small ({len(VALID_WORDS)} words). Ensure 'scrabble_words.txt' is valid. Exiting.")
        exit(1)

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
