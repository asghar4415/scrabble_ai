
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
import math

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
    # 'D' is display for D*, its score contribution is 0.
}

# --- Power Tile Definitions ---
POWER_TILE_DOUBLE_TURN_MARKER = 'D*'
# Add other markers here, e.g., POWER_TILE_BLOCK_MARKER = 'B*', POWER_TILE_WILD_MARKER = 'W*'
POWER_TILE_TYPES = {
    POWER_TILE_DOUBLE_TURN_MARKER: {"effect": "double_turn", "display": "D"}
    # Add other power tile types here:
    # POWER_TILE_BLOCK_MARKER: {"effect": "block_square", "display": "X"} # Example display
    # POWER_TILE_WILD_MARKER: {"effect": "wildcard_power", "display": "*"} # Example display
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

MAX_RACK_PERMUTATION_LENGTH = 5  # Limit permutations to 5 letters for performance
AI_THINKING_TIME_LIMIT_SECONDS = 7.0


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
        self.power_tile_effect_active = {"double_turn": False}

    def get_letter_at(self, r: int, c: int) -> Optional[str]:
        if 0 <= r < self.size and 0 <= c < self.size and self.board[r][c]:
            tile_marker = self.board[r][c][0]
            return POWER_TILE_TYPES.get(tile_marker, {}).get("display", tile_marker)
        return None

    def get_raw_tile_marker_at(self, r: int, c: int) -> Optional[str]:
        if 0 <= r < self.size and 0 <= c < self.size and self.board[r][c]:
            return self.board[r][c][0]
        return None

    def initialize_tile_bag(self) -> List[str]:
        distribution = {
            'A': 9, 'B': 2, 'C': 2, 'D': 4, 'E': 12, 'F': 2, 'G': 3, 'H': 2,
            'I': 9, 'J': 1, 'K': 1, 'L': 4, 'M': 2, 'N': 6, 'O': 8, 'P': 2,
            'Q': 1, 'R': 6, 'S': 4, 'T': 6, 'U': 4, 'V': 2, 'W': 2, 'X': 1,
            'Y': 2, 'Z': 1, ' ': 2
        }
        # Add 1 Double Turn tile
        power_tiles_to_add = {POWER_TILE_DOUBLE_TURN_MARKER: 1}
        # Add other power tiles here:
        # power_tiles_to_add[POWER_TILE_BLOCK_MARKER] = 2
        # power_tiles_to_add[POWER_TILE_WILD_MARKER] = 1

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
        # ... (implementation as before) ...
        drawn = []
        for _ in range(count):
            if not self.tile_bag:
                break
            drawn.append(self.tile_bag.pop())
        return drawn

    def initialize_premium_squares(self) -> Dict[Tuple[int, int], str]:
        # ... (implementation as before) ...
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
        for r_orig, c_orig in tw_coords:
            for mr, mc in mirror(r_orig, c_orig):
                if (mr, mc) in all_coords:
                    premiums[(mr, mc)] = 'TW'
        for r_orig, c_orig in dw_coords:
            for mr, mc in mirror(r_orig, c_orig):
                if (mr, mc) in all_coords:
                    premiums[(mr, mc)] = 'DW'
        for r_orig, c_orig in tl_coords:
            for mr, mc in mirror(r_orig, c_orig):
                if (mr, mc) in all_coords:
                    premiums[(mr, mc)] = 'TL'
        for r_orig, c_orig in dl_coords:
            for mr, mc in mirror(r_orig, c_orig):
                if (mr, mc) in all_coords:
                    premiums[(mr, mc)] = 'DL'
        premiums[(center, center)] = 'DW'
        return premiums

    def assign_objective(self) -> Dict[str, Any]:
        objective = random.choice(OBJECTIVE_TYPES)
        return {"id": objective["id"], "desc": objective["desc"], "bonus": objective["bonus"], "completed": False}

    def calculate_move_score(self, word: str, row: int, col: int, direction: str,
                             placed_tiles_info: List[Dict[str, Any]]) -> Tuple[int, List[str]]:
        # ... (implementation as before, ensuring power tile scores are 0) ...
        total_score = 0
        words_formed = []
        main_word_score = 0
        main_word_multiplier = 1
        placed_positions = {info['pos'] for info in placed_tiles_info}

        for i, letter in enumerate(word.upper()):
            r_current = row + (i if direction == "vertical" else 0)
            c_current = col + (i if direction == "horizontal" else 0)
            current_pos = (r_current, c_current)
            square_type = self.premium_squares.get(current_pos)
            is_newly_placed = current_pos in placed_positions

            letter_value = LETTER_SCORES.get(
                letter, 0)  # Score from display letter
            letter_multiplier = 1

            if is_newly_placed:
                # Check if it's a blank or a power tile from the placed_info
                tile_info_for_pos = next(
                    (info for info in placed_tiles_info if info['pos'] == current_pos), None)
                if tile_info_for_pos:
                    if tile_info_for_pos['is_blank']:
                        letter_value = 0
                    if tile_info_for_pos['tile_marker'] in POWER_TILE_TYPES:
                        letter_value = 0  # Power tiles themselves usually don't add score

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
            placed_letter = info['letter']

            cross_word_list = [placed_letter]
            cross_word_start_pos = (r_placed, c_placed)
            # Extend backwards
            cr, cc = r_placed, c_placed
            while True:
                nr, nc = (
                    cr - 1, cc) if cross_direction == "vertical" else (cr, cc - 1)
                if not (0 <= nr < self.size and 0 <= nc < self.size):
                    break
                letter_to_add = self.get_letter_at(nr, nc)
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
                letter_to_add = self.get_letter_at(nr, nc)
                if letter_to_add:
                    cross_word_list.append(letter_to_add)
                    cr, cc = nr, nc
                else:
                    break

            cross_word = "".join(cross_word_list)
            if len(cross_word) >= 2 and is_valid_word(cross_word) and cross_word not in words_formed:
                words_formed.append(cross_word)
                cross_score = 0
                cross_word_multiplier = 1
                for i_cross, letter_cross in enumerate(cross_word.upper()):
                    r_cross = cross_word_start_pos[0] + \
                        (i_cross if cross_direction == "vertical" else 0)
                    c_cross = cross_word_start_pos[1] + \
                        (i_cross if cross_direction == "horizontal" else 0)
                    current_cross_pos = (r_cross, c_cross)
                    square_type_cross = self.premium_squares.get(
                        current_cross_pos)

                    letter_value_cross = LETTER_SCORES.get(letter_cross, 0)
                    tile_info_for_cross_pos = next(
                        (p_info for p_info in placed_tiles_info if p_info['pos'] == current_cross_pos), None)
                    if tile_info_for_cross_pos:
                        if tile_info_for_cross_pos['is_blank'] or tile_info_for_cross_pos['tile_marker'] in POWER_TILE_TYPES:
                            letter_value_cross = 0

                    letter_multiplier_cross = 1
                    # Only apply premium if anchor tile is on it
                    if current_cross_pos == (r_placed, c_placed):
                        if square_type_cross == 'DL':
                            letter_multiplier_cross = 2
                        elif square_type_cross == 'TL':
                            letter_multiplier_cross = 3
                        if square_type_cross == 'DW':
                            cross_word_multiplier *= 2
                        elif square_type_cross == 'TW':
                            cross_word_multiplier *= 3
                    cross_score += letter_value_cross * letter_multiplier_cross
                total_score += cross_score * cross_word_multiplier
        if len(placed_positions) == 7:
            total_score += 50
        return total_score, sorted(list(set(words_formed)))

    def _reconstruct_full_line(self, start_row: int, start_col: int, direction: str,
                               length: int, temp_board_view: List[List[Optional[Tuple[str, bool]]]]) -> str:
        """
        Helper to reconstruct the full string of letters along a line,
        given a starting point, direction, and length of the PLACED segment,
        using a temporary board view. It extends to include all contiguous existing letters.
        temp_board_view is what the board would look like WITH the new tiles placed.
        """
        parts = []
        if direction == "horizontal":
            # Find the true start of the word on this row
            true_start_c = start_col
            while true_start_c > 0 and temp_board_view[start_row][true_start_c - 1]:
                true_start_c -= 1

            # Find the true end of the word on this row
            true_end_c = start_col + length - 1  # Last col of the placed segment
            while true_end_c < self.size - 1 and temp_board_view[start_row][true_end_c + 1]:
                true_end_c += 1

            for c_idx in range(true_start_c, true_end_c + 1):
                tile_info = temp_board_view[start_row][c_idx]
                if tile_info:  # Should always be true for a contiguous word
                    # tile_info[0] is the display letter already placed on temp_board_view
                    parts.append(tile_info[0])
                else:  # Should not happen in a validly constructed line
                    logger.error(
                        f"Gap found during full line reconstruction at ({start_row},{c_idx}) for horizontal word.")
                    return "INVALID_LINE"

        else:  # Vertical
            true_start_r = start_row
            while true_start_r > 0 and temp_board_view[true_start_r - 1][start_col]:
                true_start_r -= 1

            true_end_r = start_row + length - 1  # Last row of the placed segment
            while true_end_r < self.size - 1 and temp_board_view[true_end_r + 1][start_col]:
                true_end_r += 1

            for r_idx in range(true_start_r, true_end_r + 1):
                tile_info = temp_board_view[r_idx][start_col]
                if tile_info:
                    parts.append(tile_info[0])
                else:
                    logger.error(
                        f"Gap found during full line reconstruction at ({r_idx},{start_col}) for vertical word.")
                    return "INVALID_LINE"

        return "".join(parts)

    def validate_move(self, word_proposal: str, row: int, col: int, direction: str, player: str) -> Tuple[bool, str, List[Dict[str, Any]]]:
        rack = self.player_racks[player]
        letters_needed_from_rack = []
        connected_to_existing = self.first_move
        center_square = (self.size // 2, self.size // 2)
        touches_center = False
        num_new_tiles = 0
        word_proposal_upper = word_proposal.upper()
        temp_board_for_crossword_check = deepcopy(self.board)
        placed_tiles_info = []

        if not word_proposal or not all(c.isalnum() or c == ' ' for c in word_proposal_upper):
            return False, "Word proposal invalid.", []

        word_len = len(word_proposal_upper)

        if direction == "horizontal":
            if col < 0 or col + word_len > self.size or row < 0 or row >= self.size:
                return False, "Out of bounds.", []
        else:
            if row < 0 or row + word_len > self.size or col < 0 or col >= self.size:
                return False, "Out of bounds.", []

        for i, proposed_char_from_rack_segment in enumerate(word_proposal_upper):
            r_current = row + (i if direction == "vertical" else 0)
            c_current = col + (i if direction == "horizontal" else 0)
            pos = (r_current, c_current)

            existing_tile_marker_on_main_board = self.get_raw_tile_marker_at(
                r_current, c_current)

            if existing_tile_marker_on_main_board:  # Overlap with an existing tile
                existing_display_letter = POWER_TILE_TYPES.get(existing_tile_marker_on_main_board, {
                }).get("display", existing_tile_marker_on_main_board)
                # If AI proposes "X" but board has "Y" at overlap
                if existing_display_letter != proposed_char_from_rack_segment:
                    return False, f"Conflict at ({r_current},{c_current}): Board has '{existing_display_letter}', proposed '{proposed_char_from_rack_segment}'.", []
                # For temp_board_for_crossword_check, this square is already occupied with the correct letter
                # No new tile info needed for this specific square in placed_tiles_info, as it's not new
                connected_to_existing = True
            else:  # Placing a new tile from the rack segment
                num_new_tiles += 1
                letters_needed_from_rack.append(
                    proposed_char_from_rack_segment)
                placed_tiles_info.append({'letter': proposed_char_from_rack_segment,
                                          'pos': pos, 'is_blank': False, 'tile_marker': '?'})  # tile_marker filled later
                # Update temp_board_for_crossword_check with this newly placed tile (using its display letter for now)
                temp_board_for_crossword_check[r_current][c_current] = (
                    proposed_char_from_rack_segment, True)  # True = newly placed this turn

                if not connected_to_existing:
                    for dr_adj, dc_adj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr_adj, nc_adj = r_current + dr_adj, c_current + dc_adj
                        if 0 <= nr_adj < self.size and 0 <= nc_adj < self.size and self.get_raw_tile_marker_at(nr_adj, nc_adj):
                            connected_to_existing = True
                            break

            if self.first_move and pos == center_square:
                touches_center = True

        if num_new_tiles == 0:
            return False, "Must place at least one new tile.", []

        # ---- Step 2: Rack Validation (same as before, resolves tile_marker and is_blank in placed_tiles_info) ----
        rack_copy = list(rack)
        final_resolved_placed_tiles_info = []  # Renamed for clarity
        # ... (The rack validation logic that populates final_resolved_placed_tiles_info with correct tile_marker and is_blank)
        # --- Rack Validation: Match proposed letters to rack tiles ('A', ' ', 'D*') ---
        possible_rack_use = True
        # placed_tiles_info currently holds proposals
        unassigned_proposals = list(placed_tiles_info)

        for i in range(len(unassigned_proposals)):
            needed_display_char = unassigned_proposals[i]['letter']
            original_pos = unassigned_proposals[i]['pos']
            found_rack_tile = False

            # 1. Try exact letter match (if it's a letter)
            if needed_display_char.isalpha() and needed_display_char in rack_copy:
                rack_copy.remove(needed_display_char)
                final_resolved_placed_tiles_info.append(
                    {'letter': needed_display_char, 'pos': original_pos, 'is_blank': False, 'tile_marker': needed_display_char})
                found_rack_tile = True
            # 2. Try matching to a Power Tile display letter
            elif not found_rack_tile:
                for pt_marker, pt_data in POWER_TILE_TYPES.items():
                    if pt_data.get("display") == needed_display_char and pt_marker in rack_copy:
                        rack_copy.remove(pt_marker)
                        final_resolved_placed_tiles_info.append(
                            {'letter': needed_display_char, 'pos': original_pos, 'is_blank': False, 'tile_marker': pt_marker})
                        found_rack_tile = True
                        break
            # 3. Try using a blank tile (if it's a letter needed)
            if not found_rack_tile and needed_display_char.isalpha() and ' ' in rack_copy:
                rack_copy.remove(' ')
                final_resolved_placed_tiles_info.append(
                    {'letter': needed_display_char, 'pos': original_pos, 'is_blank': True, 'tile_marker': ' '})
                found_rack_tile = True

            if not found_rack_tile:
                possible_rack_use = False
                break

        if not possible_rack_use:
            return False, f"Not enough tiles in rack for proposal.", []
        # Now contains resolved tile info
        placed_tiles_info = final_resolved_placed_tiles_info

        # ---- Step 3: Update temp_board_for_crossword_check with actual tile markers and existing board tiles ----
        # This board view will have *all* tiles that would be part of the final words.
        # Start fresh from actual board
        final_line_temp_board = deepcopy(self.board)
        for info in placed_tiles_info:  # Place new tiles from rack
            final_line_temp_board[info['pos'][0]][info['pos'][1]] = (
                info['letter'], True)  # Use display letter

        # ---- Step 4: Game Rule Checks ----
        if self.first_move and not touches_center:
            return False, "First move must touch center.", []
        if not self.first_move and not connected_to_existing:
            return False, "Must connect to existing tiles.", []

        # ---- Step 5: Main Word Validation (using the full reconstructed line) ----
        # `row`, `col` are the start of the *placed segment*. `word_len` is its length.
        full_main_line_word = self._reconstruct_full_line(
            row, col, direction, word_len, final_line_temp_board)

        if full_main_line_word == "INVALID_LINE" or not is_valid_word(full_main_line_word):
            return False, f"Main line word '{full_main_line_word}' is not valid.", []

        # ---- Step 6: Crossword Validation ----
        # Crosswords are formed by *each newly placed tile*.
        # Use final_line_temp_board for checking crosswords too, as it has the complete picture.
        cross_dir_check = "vertical" if direction == "horizontal" else "horizontal"
        for new_tile_info in placed_tiles_info:  # Iterate only over newly placed tiles
            r_placed, c_placed = new_tile_info['pos']

            # Reconstruct cross word using final_line_temp_board
            # Start with the newly placed tile
            cross_word_parts = [new_tile_info['letter']]

            # Extend backwards (cross direction)
            current_r, current_c = r_placed, c_placed
            while True:
                next_r = current_r - 1 if cross_dir_check == "vertical" else current_r
                next_c = current_c - 1 if cross_dir_check == "horizontal" else current_c
                if not (0 <= next_r < self.size and 0 <= next_c < self.size):
                    break
                tile_on_final_board = final_line_temp_board[next_r][next_c]
                if tile_on_final_board:
                    # tile_on_final_board[0] is display letter
                    cross_word_parts.insert(0, tile_on_final_board[0])
                    current_r, current_c = next_r, next_c
                else:
                    break

            # Extend forwards (cross direction)
            current_r, current_c = r_placed, c_placed
            while True:
                next_r = current_r + 1 if cross_dir_check == "vertical" else current_r
                next_c = current_c + 1 if cross_dir_check == "horizontal" else current_c
                if not (0 <= next_r < self.size and 0 <= next_c < self.size):
                    break
                tile_on_final_board = final_line_temp_board[next_r][next_c]
                if tile_on_final_board:
                    cross_word_parts.append(tile_on_final_board[0])
                    current_r, current_c = next_r, next_c
                else:
                    break

            cross_word_formed = "".join(cross_word_parts)
            if len(cross_word_formed) >= 2 and not is_valid_word(cross_word_formed):
                return False, f"Creates invalid crossword '{cross_word_formed}' at ({r_placed},{c_placed}).", []

        return True, f"Move forming '{full_main_line_word}' is valid.", placed_tiles_info

    def place_word(self, word: str, row: int, col: int, direction: str, player: str) -> Tuple[bool, str, Optional[int]]:
        # ... (implementation as before, handles objectives and power tile effects) ...
        is_valid, message, placed_tiles_info = self.validate_move(
            word, row, col, direction, player)
        if not is_valid:
            return False, message, None

        formed_word = self._get_formed_word_from_placement(
            row, col, direction, placed_tiles_info)
        start_r_score, start_c_score = self._get_word_start(
            row, col, direction)
        score, words_formed = self.calculate_move_score(
            formed_word, start_r_score, start_c_score, direction, placed_tiles_info)

        objective_bonus = 0
        objective_msg = ""
        objective = self.player_objectives[player]
        if not objective["completed"]:
            if self.check_objective_completion(objective["id"], score, words_formed, placed_tiles_info):
                objective["completed"] = True
                objective_bonus = objective["bonus"]
                score += objective_bonus
                objective_msg = f" Objective '{objective['desc']}' (+{objective_bonus} pts)!"

        letters_removed_from_rack = []
        power_tile_triggered = None
        for info in placed_tiles_info:
            r_place, c_place = info['pos']
            tile_marker = info['tile_marker']
            self.board[r_place][c_place] = (tile_marker, True)
            letters_removed_from_rack.append(tile_marker)
            if tile_marker in POWER_TILE_TYPES:
                effect = POWER_TILE_TYPES[tile_marker].get("effect")
                if effect == "double_turn":
                    self.power_tile_effect_active["double_turn"] = True
                    power_tile_triggered = "Double Turn"

        self.scores[player] += score
        self.remove_from_rack(letters_removed_from_rack, player)
        self.refill_rack(player)
        self.first_move = False
        self.consecutive_passes = 0
        self.last_move_time = time.time()

        if not self.tile_bag and not self.player_racks[player]:
            self.game_over = True
            self.finalize_scores()

        if not self.game_over:
            if self.power_tile_effect_active["double_turn"]:
                self.power_tile_effect_active["double_turn"] = False
            else:
                self.switch_turn()

        success_message = f"Played '{formed_word}' for {score} pts (Base: {score - objective_bonus}). Words: {', '.join(words_formed)}."
        if objective_msg:
            success_message += objective_msg
        if power_tile_triggered:
            success_message += f" Triggered {power_tile_triggered}!"
        return True, success_message, score

    def _get_formed_word_from_placement(self, row: int, col: int, direction: str, placed_info: List[Dict[str, Any]]) -> str:
        # ... (implementation as before) ...
        parts = []
        placed_map = {p['pos']: p['letter'] for p in placed_info}
        if direction == "horizontal":
            start_c = col
            while start_c > 0 and self.get_raw_tile_marker_at(row, start_c - 1):
                start_c -= 1
            end_c = col
            while end_c < self.size - 1:  # Iterate to find the end of the word
                if (row, end_c + 1) in placed_map or self.get_raw_tile_marker_at(row, end_c + 1):
                    end_c += 1
                else:
                    break
            for c_idx in range(start_c, end_c + 1):
                placed = placed_map.get((row, c_idx))
                existing = self.get_letter_at(row, c_idx)
                if placed:
                    parts.append(placed)
                elif existing:
                    parts.append(existing)
        else:  # Vertical
            start_r = row
            while start_r > 0 and self.get_raw_tile_marker_at(start_r - 1, col):
                start_r -= 1
            end_r = row
            while end_r < self.size - 1:
                if (end_r + 1, col) in placed_map or self.get_raw_tile_marker_at(end_r + 1, col):
                    end_r += 1
                else:
                    break
            for r_idx in range(start_r, end_r + 1):
                placed = placed_map.get((r_idx, col))
                existing = self.get_letter_at(r_idx, col)
                if placed:
                    parts.append(placed)
                elif existing:
                    parts.append(existing)
        return "".join(parts)

    def _get_word_start(self, r_place: int, c_place: int, direction: str) -> Tuple[int, int]:
        # ... (implementation as before) ...
        start_r, start_c = r_place, c_place
        if direction == "horizontal":
            while start_c > 0 and self.get_raw_tile_marker_at(start_r, start_c - 1):
                start_c -= 1
        else:
            while start_r > 0 and self.get_raw_tile_marker_at(start_r - 1, start_c):
                start_r -= 1
        return start_r, start_c

    def check_objective_completion(self, objective_id: str, move_score: int, words_formed: List[str], placed_tiles_info: List[Dict[str, Any]]) -> bool:
        # ... (implementation as before) ...
        if objective_id == "score_gt_30":
            return move_score >= 30
        if objective_id == "use_q_z_x_j":
            return any(c in 'QZXJ' for word in words_formed for c in word)
        if objective_id == "form_7_letter":
            return len(placed_tiles_info) == 7
        if objective_id == "use_corner":
            corners = [(0, 0), (0, self.size-1), (self.size-1, 0),
                       (self.size-1, self.size-1)]
            return any(info['pos'] in corners for info in placed_tiles_info)
        return False

    def pass_turn(self, player: str) -> Tuple[bool, str]:
        # ... (implementation as before) ...
        if player != self.current_player:
            return False, "Not your turn."
        self.consecutive_passes += 1
        if self.consecutive_passes >= 6:
            self.game_over = True
            self.finalize_scores()
            return True, "Turn passed. Game Over: 6 consecutive passes."
        else:
            self.switch_turn()
            return True, "Turn passed."

    def finalize_scores(self):
        # ... (implementation as before) ...
        empty_rack_player = None
        total_unplayed_score = 0
        for p, rack in self.player_racks.items():
            player_unplayed_score = sum(LETTER_SCORES.get(
                POWER_TILE_TYPES.get(tile, {}).get("display", tile), 0) for tile in rack)
            if not rack:
                empty_rack_player = p
            else:
                self.scores[p] -= player_unplayed_score
                total_unplayed_score += player_unplayed_score
        if empty_rack_player and self.get_opponent(empty_rack_player) in self.scores:
            self.scores[empty_rack_player] += total_unplayed_score
        logger.info(
            f"Final Scores: Human: {self.scores.get('human',0)}, AI: {self.scores.get('ai',0)}")

    def remove_from_rack(self, tile_markers: List[str], player: str):
        # ... (implementation as before) ...
        rack = self.player_racks[player]
        for marker in tile_markers:
            try:
                rack.remove(marker)
            except ValueError:
                logger.error(
                    f"CRITICAL: Failed to remove marker '{marker}' from {player}'s rack {rack}")

    def refill_rack(self, player: str):
        # ... (implementation as before) ...
        needed = 7 - len(self.player_racks[player])
        if needed > 0:
            new_tiles = self.draw_tiles(needed)
            if new_tiles:
                self.player_racks[player].extend(new_tiles)
        if not self.tile_bag:
            logger.info("Tile bag is empty.")

    def switch_turn(self):
        # ... (implementation as before) ...
        self.current_player = "ai" if self.current_player == "human" else "human"
        logger.info(f"--- Turn switched to: {self.current_player} ---")

    def get_opponent(self, player: str) -> str:
        # ... (implementation as before) ...
        return "ai" if player == "human" else "human"

    def get_state(self, hide_ai_rack=True) -> Dict[str, Any]:
        # ... (implementation as before, ensures display letters are sent) ...
        display_board = [[(POWER_TILE_TYPES.get(cell[0], {}).get(
            "display", cell[0]) if cell else None) for cell in row] for row in self.board]
        human_rack_display = [POWER_TILE_TYPES.get(marker, {}).get(
            "display", marker) for marker in self.player_racks["human"]]
        ai_rack_display = []
        if not hide_ai_rack:
            ai_rack_display = [POWER_TILE_TYPES.get(marker, {}).get(
                "display", marker) for marker in self.player_racks["ai"]]

        state: Dict[str, Any] = {
            "board": display_board, "scores": self.scores, "current_player": self.current_player,
            "racks": {"human": human_rack_display}, "game_over": self.game_over, "first_move": self.first_move,
            "tiles_in_bag": len(self.tile_bag), "human_objective": self.player_objectives["human"],
        }
        if not hide_ai_rack:
            state["racks"]["ai"] = ai_rack_display
        return state

    def get_valid_moves(self, player: str) -> List[Dict[str, Any]]:
        # ... (implementation as before) ...
        valid_moves = []
        anchor_squares = find_anchor_squares(self)
        potential_placements = generate_potential_placements_anchored(
            self, player, anchor_squares)

        for move_candidate in potential_placements:
            word, row, col, direction = move_candidate["word"], move_candidate[
                "row"], move_candidate["col"], move_candidate["direction"]
            is_valid, msg, placed_info = self.validate_move(
                word, row, col, direction, player)
            if is_valid:
                formed_word = self._get_formed_word_from_placement(
                    row, col, direction, placed_info)
                start_r_score, start_c_score = self._get_word_start(
                    row, col, direction)
                score, words_formed = self.calculate_move_score(
                    formed_word, start_r_score, start_c_score, direction, placed_info)

                objective = self.player_objectives[player]
                objective_bonus = 0
                if not objective["completed"] and self.check_objective_completion(objective["id"], score, words_formed, placed_info):
                    objective_bonus = objective["bonus"]

                valid_moves.append({
                    "word": word, "row": row, "col": col, "direction": direction,
                    "score": score + objective_bonus, "base_score": score, "objective_bonus": objective_bonus,
                    "words_formed": words_formed, "placed_info": placed_info
                })
        return valid_moves


# --------------------------
# AI Player Logic (Minimax)
# --------------------------
class AIPlayer:
    MAX_DEPTH = 1  # You already set this
    AI_PLAYER = "ai"
    HUMAN_PLAYER = "human"
    MAX_MOVES_TO_EVALUATE_AT_ROOT = 10  # You already set this
    # AI_THINKING_TIME_LIMIT_SECONDS is defined globally now

    @staticmethod
    def get_best_move(board: Board) -> Optional[Dict[str, Any]]:
        turn_start_time = time.time()
        logger.info(
            f"AI ({AIPlayer.AI_PLAYER}) starting. Rack: {board.player_racks[AIPlayer.AI_PLAYER]}. Time limit: {AI_THINKING_TIME_LIMIT_SECONDS}s"
        )

        best_move_overall = None
        best_heuristic_score_overall = -math.inf

        all_possible_moves = board.get_valid_moves(AIPlayer.AI_PLAYER)

        if not all_possible_moves:
            logger.info("AI found no valid moves.")
            return None

        all_possible_moves.sort(key=lambda m: m['score'], reverse=True)

        # Fallback: best immediate scoring move if time runs out or no better Minimax result
        if all_possible_moves:
            best_move_overall = all_possible_moves[0]
            # Quickly evaluate the board state if this best immediate move was made
            temp_board_for_fallback_eval = deepcopy(board)
            # (Simplified apply for fallback evaluation)
            fb_letters_removed = [info['tile_marker']
                                  for info in best_move_overall['placed_info']]
            for info_fb in best_move_overall['placed_info']:
                temp_board_for_fallback_eval.board[info_fb['pos'][0]][info_fb['pos'][1]] = (
                    info_fb['tile_marker'], True)
            temp_board_for_fallback_eval.scores[AIPlayer.AI_PLAYER] += best_move_overall['base_score']
            if best_move_overall['objective_bonus'] > 0:
                temp_board_for_fallback_eval.scores[AIPlayer.AI_PLAYER] += best_move_overall['objective_bonus']
                temp_board_for_fallback_eval.player_objectives[AIPlayer.AI_PLAYER]['completed'] = True
            temp_board_for_fallback_eval.remove_from_rack(
                fb_letters_removed, AIPlayer.AI_PLAYER)
            temp_board_for_fallback_eval.refill_rack(AIPlayer.AI_PLAYER)
            # Important: Set next player for correct heuristic evaluation
            fb_is_double_turn = any(
                info_fb['tile_marker'] == POWER_TILE_DOUBLE_TURN_MARKER for info_fb in best_move_overall['placed_info'])
            if not fb_is_double_turn:
                temp_board_for_fallback_eval.current_player = AIPlayer.HUMAN_PLAYER
            else:
                temp_board_for_fallback_eval.current_player = AIPlayer.AI_PLAYER
            best_heuristic_score_overall = AIPlayer.evaluate_board(
                temp_board_for_fallback_eval)

        moves_for_minimax_eval = all_possible_moves[:
                                                    AIPlayer.MAX_MOVES_TO_EVALUATE_AT_ROOT]

        logger.info(
            f"AI has {len(all_possible_moves)} valid moves. Evaluating top {len(moves_for_minimax_eval)} with Minimax (Depth {AIPlayer.MAX_DEPTH})."
        )

        for idx, current_move_candidate in enumerate(moves_for_minimax_eval):
            if time.time() - turn_start_time > AI_THINKING_TIME_LIMIT_SECONDS:
                logger.warning(
                    f"AI reached time limit after evaluating {idx} moves. Returning best found so far.")
                break

            temp_board_for_sim = deepcopy(board)
            # ... (simulation logic from your previous code, ensure it's accurate)
            sim_letters_removed = []
            for info in current_move_candidate['placed_info']:
                temp_board_for_sim.board[info['pos'][0]][info['pos'][1]] = (
                    info['tile_marker'], True)
                sim_letters_removed.append(info['tile_marker'])
            temp_board_for_sim.scores[AIPlayer.AI_PLAYER] += current_move_candidate['base_score']
            if current_move_candidate['objective_bonus'] > 0:
                temp_board_for_sim.scores[AIPlayer.AI_PLAYER] += current_move_candidate['objective_bonus']
                temp_board_for_sim.player_objectives[AIPlayer.AI_PLAYER]['completed'] = True
            temp_board_for_sim.remove_from_rack(
                sim_letters_removed, AIPlayer.AI_PLAYER)
            temp_board_for_sim.refill_rack(AIPlayer.AI_PLAYER)
            sim_power_tile_double_turn = any(
                info['tile_marker'] == POWER_TILE_DOUBLE_TURN_MARKER for info in current_move_candidate['placed_info'])
            # If AI double turn, it maximizes again
            next_player_is_maximizing_in_sim = sim_power_tile_double_turn
            if not sim_power_tile_double_turn:
                temp_board_for_sim.current_player = AIPlayer.HUMAN_PLAYER
            else:
                temp_board_for_sim.current_player = AIPlayer.AI_PLAYER  # AI plays again
            temp_board_for_sim.first_move = False
            temp_board_for_sim.consecutive_passes = 0
            # --- End of simulation logic ---

            heuristic_eval_for_this_move = AIPlayer.minimax(
                temp_board_for_sim, AIPlayer.MAX_DEPTH - 1, -
                math.inf, math.inf, next_player_is_maximizing_in_sim
            )

            logger.debug(
                f"  Move {idx+1}/{len(moves_for_minimax_eval)}: {current_move_candidate['word']}. BaseScore: {current_move_candidate['score']}. Heuristic: {heuristic_eval_for_this_move}"
            )

            if heuristic_eval_for_this_move > best_heuristic_score_overall:
                best_heuristic_score_overall = heuristic_eval_for_this_move
                best_move_overall = current_move_candidate

        total_evaluation_time = time.time() - turn_start_time
        logger.info(
            f"AI evaluation loop completed in {total_evaluation_time:.3f} seconds.")

        if best_move_overall:
            logger.info(
                f"AI Final Choice: {best_move_overall['word']} @({best_move_overall['row']},{best_move_overall['col']}) "
                f"Dir: {best_move_overall['direction']}. Initial Score: {best_move_overall['score']}. "
                f"Projected Heuristic: {best_heuristic_score_overall}."
            )
            return {"word": best_move_overall["word"], "row": best_move_overall["row"], "col": best_move_overall["col"], "direction": best_move_overall["direction"]}
        else:
            # This case should ideally not be reached if all_possible_moves was non-empty,
            # as best_move_overall is initialized. If it is, it means no moves were valid.
            logger.info(
                "AI found no suitable move (or timed out before first evaluation if no initial best).")
            return None

    # ... (Minimax and evaluate_board methods remain the same as your last provided version) ...
    @staticmethod
    def minimax(board_state: Board, depth: int, alpha: float, beta: float, is_maximizing_player: bool) -> float:
        if depth == 0 or board_state.game_over:
            return AIPlayer.evaluate_board(board_state)

        current_player_for_moves = AIPlayer.AI_PLAYER if is_maximizing_player else AIPlayer.HUMAN_PLAYER
        possible_moves = board_state.get_valid_moves(current_player_for_moves)

        if not possible_moves:
            return AIPlayer.evaluate_board(board_state)

        if is_maximizing_player:
            max_eval = -math.inf
            for move in possible_moves:  # Consider limiting moves here too if depth > 0
                child_state = deepcopy(board_state)
                letters_removed = [info['tile_marker']
                                   for info in move['placed_info']]
                for info in move['placed_info']:
                    child_state.board[info['pos'][0]][info['pos'][1]] = (
                        info['tile_marker'], True)
                child_state.scores[AIPlayer.AI_PLAYER] += move['base_score']
                if move['objective_bonus'] > 0:
                    child_state.scores[AIPlayer.AI_PLAYER] += move['objective_bonus']
                    child_state.player_objectives[AIPlayer.AI_PLAYER]['completed'] = True
                child_state.remove_from_rack(
                    letters_removed, AIPlayer.AI_PLAYER)
                child_state.refill_rack(AIPlayer.AI_PLAYER)
                is_double_turn = any(
                    info['tile_marker'] == POWER_TILE_DOUBLE_TURN_MARKER for info in move['placed_info'])
                next_is_maximizing = is_double_turn
                if not is_double_turn:
                    child_state.current_player = AIPlayer.HUMAN_PLAYER
                else:
                    child_state.current_player = AIPlayer.AI_PLAYER  # AI continues if double turn
                child_state.first_move = False
                child_state.consecutive_passes = 0
                eval_score = AIPlayer.minimax(
                    child_state, depth - 1, alpha, beta, next_is_maximizing)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = math.inf
            for move in possible_moves:  # Consider limiting moves here too
                child_state = deepcopy(board_state)
                letters_removed = [info['tile_marker']
                                   for info in move['placed_info']]
                for info in move['placed_info']:
                    child_state.board[info['pos'][0]][info['pos'][1]] = (
                        info['tile_marker'], True)
                child_state.scores[AIPlayer.HUMAN_PLAYER] += move['base_score']
                if move['objective_bonus'] > 0:
                    child_state.scores[AIPlayer.HUMAN_PLAYER] += move['objective_bonus']
                    child_state.player_objectives[AIPlayer.HUMAN_PLAYER]['completed'] = True
                child_state.remove_from_rack(
                    letters_removed, AIPlayer.HUMAN_PLAYER)
                child_state.refill_rack(AIPlayer.HUMAN_PLAYER)
                is_double_turn = any(
                    info['tile_marker'] == POWER_TILE_DOUBLE_TURN_MARKER for info in move['placed_info'])
                next_is_maximizing = not is_double_turn
                if not is_double_turn:
                    child_state.current_player = AIPlayer.AI_PLAYER
                else:
                    # Human continues if double turn
                    child_state.current_player = AIPlayer.HUMAN_PLAYER
                child_state.first_move = False
                child_state.consecutive_passes = 0
                eval_score = AIPlayer.minimax(
                    child_state, depth - 1, alpha, beta, next_is_maximizing)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval

    @staticmethod
    def evaluate_board(board_state: Board) -> float:
        ai_s = board_state.scores.get(AIPlayer.AI_PLAYER, 0)
        human_s = board_state.scores.get(AIPlayer.HUMAN_PLAYER, 0)
        score_diff = ai_s - human_s
        obj_bonus_ai = 50 if board_state.player_objectives[AIPlayer.AI_PLAYER]["completed"] else 0
        obj_penalty_human = - \
            50 if board_state.player_objectives[AIPlayer.HUMAN_PLAYER]["completed"] else 0
        rack_value = 0
        ai_rack = board_state.player_racks[AIPlayer.AI_PLAYER]
        vowels = "AEIOU"
        num_vowels = sum(1 for t in ai_rack if t in vowels)
        rack_value -= abs(num_vowels - (len(ai_rack) - num_vowels)) * 1.5
        rack_value += sum(5 for t in ai_rack if t == ' ')
        rack_value += sum(3 for t in ai_rack if t == 'S')
        rack_value += sum(LETTER_SCORES.get(t, 0) *
                          0.5 for t in ai_rack if t in "JQXZ")
        return score_diff + obj_bonus_ai + obj_penalty_human + rack_value


# --- Helper functions for Move Generation ---
def find_anchor_squares(board: Board) -> List[Tuple[int, int]]:
    # ... (your existing implementation) ...
    center_square = (board.size // 2, board.size // 2)
    if board.first_move:
        return [center_square]
    anchors = set()
    for r in range(board.size):
        for c in range(board.size):
            if board.get_raw_tile_marker_at(r, c) is None:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if 0 <= r+dr < board.size and 0 <= c+dc < board.size and board.get_raw_tile_marker_at(r+dr, c+dc):
                        anchors.add((r, c))
                        break
    return list(anchors) if anchors else [center_square]


def generate_potential_placements_anchored(board: Board, player: str, anchor_squares: List[Tuple[int, int]]) -> List[Dict[str, Any]]:
    potential_placements = []
    rack = board.player_racks[player]
    if not anchor_squares or not rack:
        return []

    # Use the new constant for MAX_PERM_LEN
    # Uses global/class const
    current_max_perm_len = min(len(rack), MAX_RACK_PERMUTATION_LENGTH)
    checked_placements = set()

    logger.debug(
        f"Generating permutations up to length: {current_max_perm_len} for rack size {len(rack)}")

    for length in range(2, current_max_perm_len + 1):
        for p_tuple in set(permutations(rack, length)):
            perm_str = "".join(p_tuple)
            for r_anchor, c_anchor in anchor_squares:  # Consider limiting anchors later if still slow
                # Horizontal
                for i in range(length):
                    start_col_h = c_anchor - i
                    start_row_h = r_anchor
                    if 0 <= start_col_h and start_col_h + length <= board.size:
                        key_h = (perm_str, start_row_h,
                                 start_col_h, "horizontal")
                        if key_h not in checked_placements:
                            potential_placements.append(
                                {"word": perm_str, "row": start_row_h, "col": start_col_h, "direction": "horizontal"})
                            checked_placements.add(key_h)
                # Vertical
                for i in range(length):
                    start_row_v = r_anchor - i
                    start_col_v = c_anchor
                    if 0 <= start_row_v and start_row_v + length <= board.size:
                        key_v = (perm_str, start_row_v,
                                 start_col_v, "vertical")
                        if key_v not in checked_placements:
                            potential_placements.append(
                                {"word": perm_str, "row": start_row_v, "col": start_col_v, "direction": "vertical"})
                            checked_placements.add(key_v)

    logger.debug(
        f"Generated {len(potential_placements)} raw placements for {player}.")
    return potential_placements

# --------------------------
# API Request/Response Models
# --------------------------


class MoveRequest(BaseModel):
    # ... (as before) ...
    word: str
    row: int
    col: int
    direction: str


class GameStateResponse(BaseModel):
    # ... (as before, includes human_objective) ...
    board: List[List[Optional[str]]]
    scores: Dict[str, int]
    current_player: str
    player_rack: List[str]
    game_over: bool
    message: Optional[str] = None
    first_move: bool
    tiles_in_bag: int
    human_objective: Optional[Dict[str, Any]] = None


# --------------------------
# Global Game Instance
# --------------------------
game_board = Board()

# --------------------------
# API Endpoints
# --------------------------


@app.get("/api/game/start", response_model=GameStateResponse, tags=["Game Flow"])
async def start_game():
    # ... (as before) ...
    global game_board
    game_board = Board()
    state = game_board.get_state()
    return GameStateResponse(**state, player_rack=state["racks"]["human"], message="New game started.")


@app.post("/api/game/move", response_model=GameStateResponse, tags=["Game Actions"])
async def player_move(move: MoveRequest):
    # ... (as before) ...
    global game_board
    player = "human"
    if game_board.game_over:
        raise HTTPException(status_code=400, detail="Game over.")
    if game_board.current_player != player:
        raise HTTPException(status_code=400, detail="Not your turn.")

    success, human_msg, _ = game_board.place_word(
        move.word, move.row, move.col, move.direction, player)
    if not success:
        state = game_board.get_state()
        return GameStateResponse(**state, player_rack=state["racks"]["human"], message=f"Invalid: {human_msg}")

    ai_msg = ""
    if not game_board.game_over and game_board.current_player == "ai":
        ai_move = AIPlayer.get_best_move(game_board)
        if ai_move:
            ai_success, msg_ai, _ = game_board.place_word(
                ai_move["word"], ai_move["row"], ai_move["col"], ai_move["direction"], "ai")
            ai_msg = msg_ai if ai_success else f"AI Error ({msg_ai}). AI Passes."
            if not ai_success:
                # Ensure pass if AI's planned move fails
                game_board.pass_turn("ai")
        else:
            _, ai_msg = game_board.pass_turn("ai")
            ai_msg = f"AI Passes. ({ai_msg})"

    final_msg = f"You: {human_msg}" + (f" || AI: {ai_msg}" if ai_msg else "")
    if game_board.game_over and "GAME OVER" not in final_msg.upper():
        final_msg += f" || GAME OVER. Final: You {game_board.scores.get('human',0)}, AI {game_board.scores.get('ai',0)}"

    final_state = game_board.get_state()
    return GameStateResponse(**final_state, player_rack=final_state["racks"]["human"], message=final_msg)


@app.post("/api/game/pass", response_model=GameStateResponse, tags=["Game Actions"])
async def player_pass():
    # ... (as before) ...
    global game_board
    player = "human"
    if game_board.game_over:
        raise HTTPException(status_code=400, detail="Game over.")
    if game_board.current_player != player:
        raise HTTPException(status_code=400, detail="Not your turn.")

    _, pass_msg = game_board.pass_turn(player)

    ai_msg = ""
    if not game_board.game_over and game_board.current_player == "ai":
        ai_move = AIPlayer.get_best_move(game_board)
        if ai_move:
            ai_success, msg_ai, _ = game_board.place_word(
                ai_move["word"], ai_move["row"], ai_move["col"], ai_move["direction"], "ai")
            ai_msg = msg_ai if ai_success else f"AI Error ({msg_ai}). AI Passes."
            if not ai_success:
                game_board.pass_turn("ai")
        else:
            _, ai_msg = game_board.pass_turn("ai")
            ai_msg = f"AI Passes. ({ai_msg})"

    final_msg = f"You: {pass_msg}" + (f" || AI: {ai_msg}" if ai_msg else "")
    if game_board.game_over and "GAME OVER" not in final_msg.upper():
        final_msg += f" || GAME OVER. Final: You {game_board.scores.get('human',0)}, AI {game_board.scores.get('ai',0)}"

    final_state = game_board.get_state()
    return GameStateResponse(**final_state, player_rack=final_state["racks"]["human"], message=final_msg)


@app.get("/api/game/state", response_model=GameStateResponse, tags=["Game Info"])
async def get_current_game_state():
    # ... (as before) ...
    global game_board
    state = game_board.get_state()
    return GameStateResponse(**state, player_rack=state["racks"]["human"], message="Current state.")

# --------------------------
# Main Execution Guard
# --------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Advanced Scrabble backend server...")
    if not VALID_WORDS or len(VALID_WORDS) < 50:
        logger.critical(
            f"Word dictionary issue ({len(VALID_WORDS)} words). Ensure 'scrabble_words.txt' is valid. Exiting.")
        exit(1)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
