# --- START OF FILE main.py ---

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

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
app = FastAPI(title="Advanced Scrabble AI Game Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173",
                   "http://127.0.0.1:5173", "https://scrabble-sandy.vercel.app"],  # Add your Vercel URL if deployed
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"], expose_headers=["*"]
)
LETTER_SCORES = {
    'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1, 'F': 4, 'G': 2, 'H': 4, 'I': 1, 'J': 8,
    'K': 5, 'L': 1, 'M': 3, 'N': 1, 'O': 1, 'P': 3, 'Q': 10, 'R': 1, 'S': 1, 'T': 1,
    'U': 1, 'V': 4, 'W': 4, 'X': 8, 'Y': 4, 'Z': 10, ' ': 0,
}
# Internal marker for the Double Turn power tile
POWER_TILE_DOUBLE_TURN_MARKER = 'D*'
POWER_TILE_TYPES = {POWER_TILE_DOUBLE_TURN_MARKER: {
    "effect": "double_turn", "display": "D"}}  # 'D' is what's shown on the board/rack
OBJECTIVE_TYPES = [
    {"id": "score_gt_30", "desc": "Score 30+ points in a single turn", "bonus": 20},
    {"id": "use_q_z_x_j", "desc": "Play a word using Q, Z, X, or J", "bonus": 15},
    {"id": "form_7_letter",
        "desc": "Form a 7-letter word (Bingo already gives 50)", "bonus": 25},  # Bingo is 7 tiles from rack
    {"id": "use_corner",
        "desc": "Play a tile on a corner square (0,0/0,14/14,0/14,14)", "bonus": 10},
]


def initialize_dictionary() -> Set[str]:
    words = set()
    # Try to find the dictionary file in common locations
    possible_paths = ['scrabble_words.txt', os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'scrabble_words.txt'), './scrabble_words.txt']
    dict_path_found = next(
        (path for path in possible_paths if os.path.exists(path)), None)

    minimal_word_set = {"QI", "ZA", "CAT", "DOG", "JO", "AX", "EX",
                        "OX", "XI", "XU", "WORD", "PLAY", "GAME", "POWER", "TURN"}

    if not dict_path_found:
        logger.warning(
            "scrabble_words.txt not found at expected paths. Using minimal word set.")
        return minimal_word_set
    try:
        with open(dict_path_found, 'r', encoding='utf-8') as f:
            # Ensure words are uppercase, at least 2 chars, and alphabetic
            loaded_words = {line.strip().upper() for line in f if len(
                line.strip()) >= 2 and line.strip().isalpha()}
        if not loaded_words:
            logger.warning(
                f"Dictionary file {dict_path_found} was empty or contained no valid words. Using minimal set.")
            words = minimal_word_set
        else:
            words = loaded_words
            logger.info(
                f"Successfully loaded {len(words)} words from {dict_path_found}")
        return words
    except Exception as e:
        logger.error(
            f"Error reading dictionary file {dict_path_found}: {e}. Using minimal word set.")
        return minimal_word_set


VALID_WORDS = initialize_dictionary()
# Max length of permutations from rack to check (performance)
MAX_RACK_PERMUTATION_LENGTH = 5
AI_THINKING_TIME_LIMIT_SECONDS = 7.0  # Time limit for AI move calculation


def is_valid_word(word: str) -> bool:
    if not word or len(word) < 2 or not word.isalpha():  # Basic checks
        return False
    return word.upper() in VALID_WORDS


class Board:
    def __init__(self):
        self.size = 15
        # Board stores (tile_marker, is_permanent_flag)
        # tile_marker can be 'A', 'B', ..., ' ', or POWER_TILE_DOUBLE_TURN_MARKER
        self.board: List[List[Optional[Tuple[str, bool]]]] = [
            [None for _ in range(self.size)] for _ in range(self.size)]
        self.scores = {"human": 0, "ai": 0}
        self.tile_bag = self.initialize_tile_bag()
        self.player_racks = {"human": self.draw_tiles(
            7), "ai": self.draw_tiles(7)}
        self.current_player = "human"  # Human always starts
        self.game_over = False
        self.first_move = True
        self.premium_squares = self.initialize_premium_squares()
        self.last_move_time = time.time()  # For potential future timeout logic
        self.consecutive_passes = 0
        self.player_objectives = {  # Each player gets one objective
            "human": self.assign_objective(), "ai": self.assign_objective()}
        self.power_tile_effect_active = {"double_turn": False}

    def get_letter_at(self, r: int, c: int) -> Optional[str]:
        """Gets the display letter at a board position (e.g., 'D' for power tile)."""
        if 0 <= r < self.size and 0 <= c < self.size and self.board[r][c]:
            tile_marker = self.board[r][c][0]
            return POWER_TILE_TYPES.get(tile_marker, {}).get("display", tile_marker)
        return None

    def get_raw_tile_marker_at(self, r: int, c: int) -> Optional[str]:
        """Gets the raw tile marker (e.g., 'D*') at a board position."""
        if 0 <= r < self.size and 0 <= c < self.size and self.board[r][c]:
            return self.board[r][c][0]
        return None

    def initialize_tile_bag(self) -> List[str]:
        # Standard Scrabble tile distribution
        distribution = {'A': 9, 'B': 2, 'C': 2, 'D': 4, 'E': 12, 'F': 2, 'G': 3, 'H': 2, 'I': 9, 'J': 1, 'K': 1, 'L': 4, 'M': 2,
                        'N': 6, 'O': 8, 'P': 2, 'Q': 1, 'R': 6, 'S': 4, 'T': 6, 'U': 4, 'V': 2, 'W': 2, 'X': 1, 'Y': 2, 'Z': 1, ' ': 2}
        # Add one Double Turn tile
        power_tiles_to_add = {POWER_TILE_DOUBLE_TURN_MARKER: 1}

        tile_bag = [letter for letter, count in distribution.items()
                    for _ in range(count)]
        tile_bag.extend(pt_marker for pt_marker,
                        count in power_tiles_to_add.items() for _ in range(count))
        random.shuffle(tile_bag)
        return tile_bag

    def draw_tiles(self, count: int) -> List[str]:
        drawn = []
        bag = self.tile_bag
        for _ in range(count):
            if not bag:
                break
            drawn.append(bag.pop())
        return drawn

    def initialize_premium_squares(self) -> Dict[Tuple[int, int], str]:
        premiums = {}
        size = self.size
        center = size // 2

        # Define by quadrant and mirror
        tw_coords = [(0, 0), (0, 7), (7, 0)]  # Triple Word
        dw_coords = [(r, r) for r in range(1, 5)] + \
            [(r, size - 1 - r)
             for r in range(1, 5)]  # Double Word (diagonals, not center)
        tl_coords = [(1, 5), (1, 9), (5, 1), (5, 5),
                     (5, 9), (5, 13)]  # Triple Letter
        dl_coords = [(0, 3), (0, 11), (2, 6), (2, 8), (3, 0), (3, 7), (3, 14),  # Double Letter
                     (6, 2), (6, 6), (6, 8), (6, 12), (7, 3), (7, 11)]

        all_coords = set((r, c) for r in range(size) for c in range(size))

        def mirror_and_add(base_coords, p_type):
            for r_orig, c_orig in base_coords:
                mirrored_points = [
                    (r_orig, c_orig), (r_orig, size - 1 - c_orig),
                    (size - 1 - r_orig, c_orig), (size -
                                                  1 - r_orig, size - 1 - c_orig)
                ]
                for mr, mc in mirrored_points:
                    if (mr, mc) in all_coords:  # Ensure it's within board bounds
                        premiums[(mr, mc)] = p_type

        mirror_and_add(tw_coords, 'TW')
        mirror_and_add(dw_coords, 'DW')
        mirror_and_add(tl_coords, 'TL')
        mirror_and_add(dl_coords, 'DL')

        premiums[(center, center)] = 'DW'  # Center is always Double Word
        return premiums

    def assign_objective(self) -> Dict[str, Any]:
        obj_template = random.choice(OBJECTIVE_TYPES)
        return {"id": obj_template["id"], "desc": obj_template["desc"], "bonus": obj_template["bonus"], "completed": False}

    def calculate_move_score(self, word: str, row: int, col: int, direction: str, placed_tiles_info: List[Dict[str, Any]]) -> Tuple[int, List[str]]:
        """Calculates score for a validated move, including crosswords and bingo."""
        total_score = 0
        words_formed_this_move = []  # To store all unique words formed

        # 1. Calculate score for the main word
        main_word_score = 0
        main_word_multiplier = 1
        # Set of (r,c) positions where new tiles were placed
        newly_placed_positions = {info['pos'] for info in placed_tiles_info}

        # word is the full line word
        for i, letter_char in enumerate(word.upper()):
            current_r = row + (i if direction == "vertical" else 0)
            current_c = col + (i if direction == "horizontal" else 0)

            current_pos = (current_r, current_c)
            square_type = self.premium_squares.get(current_pos)
            is_newly_placed_tile = current_pos in newly_placed_positions

            letter_value = LETTER_SCORES.get(letter_char, 0)
            letter_multiplier_for_score = 1

            if is_newly_placed_tile:
                # Check if this tile was a blank or power tile from rack (0 base value for score)
                tile_info = next(
                    (info for info in placed_tiles_info if info['pos'] == current_pos), None)
                if tile_info and (tile_info['is_blank'] or tile_info['tile_marker'] in POWER_TILE_TYPES):
                    letter_value = 0  # Blanks and Power Tiles contribute 0 to letter score

                # Apply letter multipliers (DL, TL) only for newly placed tiles
                if square_type == 'DL':
                    letter_multiplier_for_score = 2
                elif square_type == 'TL':
                    letter_multiplier_for_score = 3

                # Accumulate word multipliers (DW, TW) also only for newly placed tiles
                if square_type == 'DW':
                    main_word_multiplier *= 2
                elif square_type == 'TW':
                    main_word_multiplier *= 3

            main_word_score += letter_value * letter_multiplier_for_score

        main_word_score *= main_word_multiplier
        total_score += main_word_score
        if word not in words_formed_this_move:
            words_formed_this_move.append(word)

        # 2. Calculate scores for any crosswords formed
        cross_direction = "vertical" if direction == "horizontal" else "horizontal"
        for placed_tile_info in placed_tiles_info:  # Iterate only over newly placed tiles
            r_placed, c_placed = placed_tile_info['pos']
            # Display letter of the placed tile
            placed_letter_char = placed_tile_info['letter']

            cross_word_parts = [placed_letter_char]
            cross_word_start_pos_r, cross_word_start_pos_c = r_placed, c_placed

            # Extend backward in cross_direction
            temp_r, temp_c = r_placed, c_placed
            while True:
                next_r = temp_r - 1 if cross_direction == "vertical" else temp_r
                next_c = temp_c - 1 if cross_direction == "horizontal" else temp_c
                if not (0 <= next_r < self.size and 0 <= next_c < self.size):
                    break

                existing_letter = self.get_letter_at(next_r, next_c)
                if existing_letter:
                    cross_word_parts.insert(0, existing_letter)
                    cross_word_start_pos_r, cross_word_start_pos_c = next_r, next_c
                    temp_r, temp_c = next_r, next_c
                else:
                    break

            # Extend forward in cross_direction
            temp_r, temp_c = r_placed, c_placed
            while True:
                next_r = temp_r + 1 if cross_direction == "vertical" else temp_r
                next_c = temp_c + 1 if cross_direction == "horizontal" else temp_c
                if not (0 <= next_r < self.size and 0 <= next_c < self.size):
                    break

                existing_letter = self.get_letter_at(next_r, next_c)
                if existing_letter:
                    cross_word_parts.append(existing_letter)
                    temp_r, temp_c = next_r, next_c
                else:
                    break

            cross_word_formed = "".join(cross_word_parts)
            if len(cross_word_formed) >= 2 and is_valid_word(cross_word_formed) and cross_word_formed not in words_formed_this_move:
                if cross_word_formed not in words_formed_this_move:
                    words_formed_this_move.append(cross_word_formed)

                current_cross_word_score = 0
                current_cross_word_multiplier = 1

                for i_cw, letter_cw_char in enumerate(cross_word_formed.upper()):
                    r_cw = cross_word_start_pos_r + \
                        (i_cw if cross_direction == "vertical" else 0)
                    c_cw = cross_word_start_pos_c + \
                        (i_cw if cross_direction == "horizontal" else 0)

                    pos_cw = (r_cw, c_cw)
                    sq_type_cw = self.premium_squares.get(pos_cw)
                    val_cw = LETTER_SCORES.get(letter_cw_char, 0)

                    # If this letter in the crossword is the one we just placed:
                    if pos_cw == (r_placed, c_placed):
                        # Use its original tile info for blank/power tile check
                        if placed_tile_info['is_blank'] or placed_tile_info['tile_marker'] in POWER_TILE_TYPES:
                            val_cw = 0

                        # Apply letter multipliers
                        letter_mult_cw = 1
                        if sq_type_cw == 'DL':
                            letter_mult_cw = 2
                        elif sq_type_cw == 'TL':
                            letter_mult_cw = 3
                        current_cross_word_score += val_cw * letter_mult_cw

                        # Apply word multipliers
                        if sq_type_cw == 'DW':
                            current_cross_word_multiplier *= 2
                        elif sq_type_cw == 'TW':
                            current_cross_word_multiplier *= 3
                    else:  # It's an existing tile on the board
                        current_cross_word_score += val_cw  # No new multipliers for existing tiles

                total_score += current_cross_word_score * current_cross_word_multiplier

        # 3. Add Bingo bonus if 7 tiles were played from the rack
        if len(newly_placed_positions) == 7:
            total_score += 50  # Bingo!

        return total_score, sorted(list(set(words_formed_this_move)))

    def _reconstruct_full_line(self, start_row: int, start_col: int, direction: str,
                               length_of_placed_segment: int,  # Length of the player's placed tiles, not full word
                               temp_board_view: List[List[Optional[Tuple[str, bool]]]]) -> Tuple[str, int, int]:
        """
        Reconstructs the full word formed along a line, including existing tiles.
        temp_board_view should represent the board *with* the proposed tiles placed (using display letters).
        Returns (full_word, actual_start_row, actual_start_col).
        """
        parts = []
        # Start of the player's segment
        actual_start_r, actual_start_c = start_row, start_col

        if direction == "horizontal":
            # Find the true start of the word horizontally
            current_c = start_col
            while current_c > 0 and temp_board_view[start_row][current_c - 1]:
                current_c -= 1
            actual_start_c = current_c  # This is the first letter of the full horizontal word

            # Read the full word
            end_c = start_col + length_of_placed_segment - 1  # Last C of player's segment
            while current_c <= end_c or (current_c < self.size and temp_board_view[start_row][current_c]):
                if current_c >= self.size:
                    break  # Boundary check
                tile_info = temp_board_view[start_row][current_c]
                if tile_info:
                    # Use display letter for power tiles, otherwise the letter itself from tile_info[0]
                    display_char = POWER_TILE_TYPES.get(
                        tile_info[0], {}).get("display", tile_info[0])
                    parts.append(display_char)
                else:  # Should not happen if validation logic is correct and temp_board_view is built right
                    if current_c <= end_c:  # Only error if within expected segment range based on player input
                        logger.error(
                            f"Gap in H-line reconstruction at ({start_row},{current_c}) based on input segment.")
                        return "INVALID_LINE", start_row, actual_start_c
                    else:
                        break  # Reached end of word beyond player's segment
                current_c += 1
        else:  # Vertical
            current_r = start_row
            while current_r > 0 and temp_board_view[current_r - 1][start_col]:
                current_r -= 1
            actual_start_r = current_r

            end_r = start_row + length_of_placed_segment - 1
            while current_r <= end_r or (current_r < self.size and temp_board_view[current_r][start_col]):
                if current_r >= self.size:
                    break
                tile_info = temp_board_view[current_r][start_col]
                if tile_info:
                    display_char = POWER_TILE_TYPES.get(
                        tile_info[0], {}).get("display", tile_info[0])
                    parts.append(display_char)
                else:
                    if current_r <= end_r:
                        logger.error(
                            f"Gap in V-line reconstruction at ({current_r},{start_col}) based on input segment.")
                        return "INVALID_LINE", actual_start_r, start_col
                    else:
                        break
                current_r += 1

        return "".join(parts), actual_start_r, actual_start_c

    def validate_move(self, word_proposal_segment: str,  # The letters the player wants to place from their rack
                      row_segment_start: int, col_segment_start: int, direction: str, player: str) -> Tuple[bool, str, List[Dict[str, Any]], str, int, int]:
        """
        Validates a proposed move.
        Returns: (isValid, message, placed_tiles_info_list, full_word_formed, full_word_start_row, full_word_start_col)
        placed_tiles_info_list: [{'letter': 'A', 'pos': (r,c), 'is_blank': False, 'tile_marker': 'A'}, ...]
        """
        rack = self.player_racks[player]
        placed_tiles_info_proposals = []  # Info about tiles from rack to be placed

        # For first move, connection is implicit to center
        connected_to_existing = self.first_move
        center_square = (self.size // 2, self.size // 2)
        touches_center_square = False
        num_new_tiles_from_rack = 0

        word_proposal_segment_upper = word_proposal_segment.upper()
        segment_len = len(word_proposal_segment_upper)

        # ' ' for blank, or 'D' for power tile display
        if not word_proposal_segment_upper or not all(c.isalnum() or c == ' ' for c in word_proposal_segment_upper):
            return False, "Word proposal format invalid (must be alphanumeric or space/power display).", [], "", 0, 0

        if segment_len == 0:
            return False, "Must place at least one tile.", [], "", 0, 0

        # 1. Bounds check for the segment to be placed
        if direction == "horizontal":
            if not (0 <= row_segment_start < self.size and 0 <= col_segment_start < self.size and 0 <= col_segment_start + segment_len - 1 < self.size):
                return False, "Placement segment out of bounds.", [], "", 0, 0
        else:  # Vertical
            if not (0 <= col_segment_start < self.size and 0 <= row_segment_start < self.size and 0 <= row_segment_start + segment_len - 1 < self.size):
                return False, "Placement segment out of bounds.", [], "", 0, 0

        # 2. Create a temporary board view including existing tiles AND the proposed new segment
        # This temp board will store DISPLAY letters for _reconstruct_full_line.
        # Stores (raw_marker, is_permanent)
        temp_board_for_line_check = deepcopy(self.board)

        current_rack_copy = list(rack)

        for i in range(segment_len):
            r_curr = row_segment_start + (i if direction == "vertical" else 0)
            c_curr = col_segment_start + \
                (i if direction == "horizontal" else 0)
            pos_curr = (r_curr, c_curr)

            # The char from player's input string for this spot
            proposed_display_char_for_pos = word_proposal_segment_upper[i]

            existing_tile_marker_on_board = self.get_raw_tile_marker_at(
                r_curr, c_curr)
            existing_display_char_on_board = self.get_letter_at(r_curr, c_curr)

            if existing_tile_marker_on_board:  # Cell on main board is already occupied
                if existing_display_char_on_board != proposed_display_char_for_pos:
                    return False, f"Conflict at ({r_curr},{c_curr}): Board has '{existing_display_char_on_board}', proposed '{proposed_display_char_for_pos}'.", [], "", 0, 0
                # If matches, this position is part of the word but uses an existing tile.
                # No new tile placed here. For temp_board_for_line_check, it's already there from deepcopy.
                connected_to_existing = True
            else:  # Cell is empty, must use a tile from rack
                num_new_tiles_from_rack += 1

                # Find the actual tile_marker from rack corresponding to proposed_display_char_for_pos
                found_rack_tile_for_char = False
                tile_marker_from_rack = ''
                is_blank_used = False

                # Prioritize exact letter match
                if proposed_display_char_for_pos.isalpha() and proposed_display_char_for_pos in current_rack_copy:
                    tile_marker_from_rack = proposed_display_char_for_pos
                    current_rack_copy.remove(proposed_display_char_for_pos)
                    found_rack_tile_for_char = True
                # Then check for power tiles by their display char
                elif not found_rack_tile_for_char:
                    for pt_m, pt_d in POWER_TILE_TYPES.items():
                        if pt_d.get("display") == proposed_display_char_for_pos and pt_m in current_rack_copy:
                            tile_marker_from_rack = pt_m
                            current_rack_copy.remove(pt_m)
                            found_rack_tile_for_char = True
                            break
                # Finally, check for blanks if it's an alphabet
                if not found_rack_tile_for_char and proposed_display_char_for_pos.isalpha() and ' ' in current_rack_copy:
                    tile_marker_from_rack = ' '
                    current_rack_copy.remove(' ')
                    is_blank_used = True
                    found_rack_tile_for_char = True

                if not found_rack_tile_for_char:
                    return False, f"Cannot form '{word_proposal_segment_upper}': Tile '{proposed_display_char_for_pos}' not available in rack {rack}.", [], "", 0, 0

                placed_tiles_info_proposals.append({
                    # The letter this tile REPRESENTS on board
                    'letter': proposed_display_char_for_pos,
                    'pos': pos_curr,
                    'is_blank': is_blank_used,
                    # The actual item from the rack ('A', ' ', 'D*')
                    'tile_marker': tile_marker_from_rack
                })
                # Add this newly placed tile (its display char) to the temp_board_for_line_check
                temp_board_for_line_check[r_curr][c_curr] = (
                    proposed_display_char_for_pos, True)  # (display_char, is_temp)

                # Check adjacency for connection if not first move
                if not self.first_move and not connected_to_existing:
                    for dr_adj, dc_adj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr_adj, nc_adj = r_curr + dr_adj, c_curr + dc_adj
                        if 0 <= nr_adj < self.size and 0 <= nc_adj < self.size and self.get_raw_tile_marker_at(nr_adj, nc_adj):
                            connected_to_existing = True
                            break  # Found connection for this new tile

            if pos_curr == center_square:
                touches_center_square = True

        # Should not happen if segment_len > 0 and all were existing, but good check.
        if num_new_tiles_from_rack == 0:
            return False, "No new tiles were placed from the rack.", [], "", 0, 0

        # 3. Game Rule Checks
        if self.first_move and not touches_center_square:
            return False, "First move must cross the center square.", [], "", 0, 0
        if not self.first_move and not connected_to_existing:
            return False, "Move must connect to existing tiles on the board.", [], "", 0, 0

        # 4. Main Word Validation (using the temp board that has new+old tiles as display chars)
        # The 'length' here is tricky. _reconstruct_full_line needs the length of the *player's placed segment*
        # to know the extent of the player's direct play.
        # The player's input `word_proposal_segment` has length `segment_len`.
        # The `row_segment_start`, `col_segment_start` are the start of this segment.
        full_main_word, full_word_start_r, full_word_start_c = self._reconstruct_full_line(
            row_segment_start, col_segment_start, direction, segment_len, temp_board_for_line_check
        )

        if full_main_word == "INVALID_LINE" or not is_valid_word(full_main_word):
            return False, f"Main word '{full_main_word}' is not valid.", [], "", 0, 0

        # 5. Crossword Validation (iterating over newly placed tiles from `placed_tiles_info_proposals`)
        cross_dir_check = "vertical" if direction == "horizontal" else "horizontal"
        for new_tile_info in placed_tiles_info_proposals:
            r_p, c_p = new_tile_info['pos']
            # Use _reconstruct_full_line for the cross direction.
            # For cross words, the "segment" length is 1 (the new tile itself).
            # The start row/col for _reconstruct_full_line is the position of the new tile.
            cross_word_str, _, _ = self._reconstruct_full_line(
                r_p, c_p, cross_dir_check, 1, temp_board_for_line_check)

            if len(cross_word_str) >= 2 and not is_valid_word(cross_word_str):
                return False, f"Creates invalid crossword '{cross_word_str}' at ({r_p},{c_p}).", [], "", 0, 0

        return True, f"Move forming '{full_main_word}' is valid.", placed_tiles_info_proposals, full_main_word, full_word_start_r, full_word_start_c

    def place_word(self, word_segment_from_player: str, row_segment_start: int, col_segment_start: int, direction: str, player: str) -> Tuple[bool, str, Optional[int]]:
        is_valid, message, resolved_placed_tiles_info, full_main_word_formed, full_word_r, full_word_c = self.validate_move(
            word_segment_from_player, row_segment_start, col_segment_start, direction, player)

        if not is_valid:
            return False, message, None

        # Calculate score using the full main word and its actual start, and resolved_placed_tiles_info
        score, words_formed_details = self.calculate_move_score(
            full_main_word_formed, full_word_r, full_word_c, direction, resolved_placed_tiles_info)

        objective_bonus = 0
        objective_msg_part = ""
        current_player_objective = self.player_objectives[player]
        if not current_player_objective["completed"]:
            if self.check_objective_completion(current_player_objective["id"], score, words_formed_details, resolved_placed_tiles_info):
                current_player_objective["completed"] = True
                objective_bonus = current_player_objective["bonus"]
                objective_msg_part = f" Objective '{current_player_objective['desc']}' completed (+{objective_bonus} pts)!"

        final_score_for_turn = score + objective_bonus

        # --- Apply changes to the actual board ---
        letters_to_remove_from_rack_actual = []
        power_tile_triggered_effect = None

        for tile_info in resolved_placed_tiles_info:  # tile_info contains 'tile_marker'
            r_place, c_place = tile_info['pos']
            actual_tile_marker = tile_info['tile_marker']  # 'A', ' ', 'D*'

            # Store raw marker, mark as permanent
            self.board[r_place][c_place] = (actual_tile_marker, True)
            letters_to_remove_from_rack_actual.append(actual_tile_marker)

            if actual_tile_marker in POWER_TILE_TYPES:
                effect = POWER_TILE_TYPES[actual_tile_marker].get("effect")
                if effect == "double_turn":
                    self.power_tile_effect_active["double_turn"] = True
                    power_tile_triggered_effect = "Double Turn"

        self.scores[player] += final_score_for_turn
        self.remove_from_rack(letters_to_remove_from_rack_actual, player)
        self.refill_rack(player)

        self.first_move = False
        self.consecutive_passes = 0
        self.last_move_time = time.time()

        # Player played last tile
        if not self.tile_bag and not self.player_racks[player]:
            self.game_over = True
            self.finalize_scores()

        if not self.game_over:
            if self.power_tile_effect_active["double_turn"]:
                # Effect consumed
                self.power_tile_effect_active["double_turn"] = False
                # Player does not switch turn
            else:
                self.switch_turn()

        success_message_main = f"Played '{full_main_word_formed}' for {final_score_for_turn} pts (Base score: {score}). Words formed: {', '.join(words_formed_details)}."
        if objective_msg_part:
            success_message_main += objective_msg_part
        if power_tile_triggered_effect:
            success_message_main += f" Triggered {power_tile_triggered_effect}!"

        return True, success_message_main, final_score_for_turn

    def _get_word_start(self, r_start_segment: int, c_start_segment: int, direction: str) -> Tuple[int, int]:
        # This is used by AI to find the true start of a word line it might play along.
        # For place_word, validate_move now returns the full word start.
        r, c = r_start_segment, c_start_segment
        if direction == "horizontal":
            while c > 0 and self.get_raw_tile_marker_at(r, c - 1):
                c -= 1
        else:  # vertical
            while r > 0 and self.get_raw_tile_marker_at(r - 1, c):
                r -= 1
        return r, c

    def check_objective_completion(self, objective_id: str, current_turn_score: int, words_this_turn: List[str], placed_tiles_info: List[Dict[str, Any]]) -> bool:
        if objective_id == "score_gt_30":
            return current_turn_score >= 30  # Base score for the turn
        if objective_id == "use_q_z_x_j":
            high_value_letters = {'Q', 'Z', 'X', 'J'}
            for tile_info in placed_tiles_info:  # Check only newly placed tiles
                if tile_info['letter'] in high_value_letters:
                    return True
            return False
        if objective_id == "form_7_letter":  # Check if any word formed this turn is 7+ letters
            return any(len(word) >= 7 for word in words_this_turn)
        if objective_id == "use_corner":
            corners = {(0, 0), (0, self.size-1), (self.size-1, 0),
                       (self.size-1, self.size-1)}
            for tile_info in placed_tiles_info:
                if tile_info['pos'] in corners:
                    return True
            return False
        return False

    def pass_turn(self, player: str) -> Tuple[bool, str]:
        self.consecutive_passes += 1
        if self.consecutive_passes >= 2 * 2:  # 2 players, 2 passes each = game over
            self.game_over = True
            self.finalize_scores()
            return True, f"{player} passed. Game over due to consecutive passes."

        self.switch_turn()
        return True, f"{player} passed turn."

    def finalize_scores(self):
        # Deduct scores for remaining tiles for players with tiles
        # Add scores to players who emptied their rack (if applicable)
        player_who_emptied_rack = None
        for p_check in ["human", "ai"]:
            if not self.player_racks[p_check]:
                player_who_emptied_rack = p_check
                break

        if player_who_emptied_rack:
            opponent_key = self.get_opponent(player_who_emptied_rack)
            opponent_remaining_rack_value = 0
            for tile_marker in self.player_racks[opponent_key]:
                # Blanks and power tiles don't add/subtract value at end game
                if tile_marker != ' ' and tile_marker not in POWER_TILE_TYPES:
                    opponent_remaining_rack_value += LETTER_SCORES.get(
                        tile_marker, 0)

            self.scores[player_who_emptied_rack] += opponent_remaining_rack_value
            # Opponent loses value of their own tiles
            self.scores[opponent_key] -= opponent_remaining_rack_value
        else:  # No one emptied rack (e.g. game ended by passes with tiles left for both)
            for p_final in ["human", "ai"]:
                player_rack_value = 0
                for tile_marker in self.player_racks[p_final]:
                    if tile_marker != ' ' and tile_marker not in POWER_TILE_TYPES:
                        player_rack_value += LETTER_SCORES.get(tile_marker, 0)
                self.scores[p_final] -= player_rack_value

    def remove_from_rack(self, tile_markers_to_remove: List[str], player: str):
        rack = self.player_racks[player]
        for tile_marker in tile_markers_to_remove:
            if tile_marker in rack:
                rack.remove(tile_marker)
            else:  # Should not happen if validation is correct
                logger.error(
                    f"Attempted to remove non-existent tile '{tile_marker}' from {player}'s rack {rack}")

    def refill_rack(self, player: str):
        needed = 7 - len(self.player_racks[player])
        if needed > 0:
            drawn_tiles = self.draw_tiles(needed)
            self.player_racks[player].extend(drawn_tiles)

    def switch_turn(self):
        self.current_player = "ai" if self.current_player == "human" else "human"

    def get_opponent(self, player: str) -> str:
        return "ai" if player == "human" else "human"

    def get_state(self) -> Dict[str, Any]:
        display_board = []
        for r in range(self.size):
            row_display = []
            for c in range(self.size):
                tile_marker_on_board = self.get_raw_tile_marker_at(r, c)
                if tile_marker_on_board:
                    # Use display letter for power tiles, otherwise the letter itself
                    display_letter = POWER_TILE_TYPES.get(
                        tile_marker_on_board, {}).get("display", tile_marker_on_board)
                    row_display.append(display_letter)
                else:
                    row_display.append(None)
            display_board.append(row_display)

        # For player_rack sent to frontend, convert raw markers to display chars
        human_rack_display = [POWER_TILE_TYPES.get(tile, {}).get(
            "display", tile) for tile in self.player_racks["human"]]

        return {
            "board": display_board,
            "scores": self.scores.copy(),
            "current_player": self.current_player,
            # Always human's rack (display chars) for frontend
            "player_rack": human_rack_display,
            "game_over": self.game_over,
            "first_move": self.first_move,
            "tiles_in_bag": len(self.tile_bag),
            # Always human's objective
            "human_objective": self.player_objectives.get("human"),
            # Internal state if needed by AI, not directly by GameStateResponse model
            "internal_racks": self.player_racks.copy(),  # AI needs raw markers
            "internal_player_objectives": self.player_objectives.copy()
        }

    def get_valid_moves(self, player: str) -> List[Dict[str, Any]]:
        """Generates all valid moves for the given player."""
        valid_moves = []
        anchor_sqs = find_anchor_squares(self)
        # generate_potential_placements_anchored uses the raw rack for permutations
        # but returns "word" as display characters for validate_move input
        potential_placements_player_input_segments = generate_potential_placements_anchored(
            self, player, anchor_sqs)

        for placement_input in potential_placements_player_input_segments:
            # `word` here is the segment of tiles (display chars) player intends to play
            word_segment_display_chars = placement_input["word"]
            row = placement_input["row"]
            col = placement_input["col"]
            direction = placement_input["direction"]

            is_move_valid, _, placed_tiles_info_list, full_word, full_word_r, full_word_c = self.validate_move(
                word_segment_display_chars, row, col, direction, player
            )

            if is_move_valid:
                base_score, _ = self.calculate_move_score(
                    full_word, full_word_r, full_word_c, direction, placed_tiles_info_list)

                # Check objective completion for this potential move without modifying board state
                objective_bonus_for_move = 0
                # Use player_objectives from Board instance
                player_obj = self.player_objectives[player]
                if not player_obj["completed"]:  # Check only if not already completed
                    # For calculating potential objective bonus, need words_formed by this move
                    _, temp_words_formed = self.calculate_move_score(
                        full_word, full_word_r, full_word_c, direction, placed_tiles_info_list)
                    if self.check_objective_completion(player_obj["id"], base_score, temp_words_formed, placed_tiles_info_list):
                        objective_bonus_for_move = player_obj["bonus"]

                total_potential_score = base_score + objective_bonus_for_move

                valid_moves.append({
                    "word": word_segment_display_chars,  # The display chars of tiles player places
                    "row": row, "col": col, "direction": direction,
                    "score": total_potential_score,  # Full score including potential objective
                    "base_score": base_score,  # Score from words only
                    "objective_bonus": objective_bonus_for_move,
                    # Crucial for AI to simulate ('tile_marker' is raw)
                    "placed_info": placed_tiles_info_list,
                    "full_word_played": full_word  # The actual word formed on the line
                })
        return valid_moves


class AIPlayer:
    MAX_DEPTH = 1  # Adjust for difficulty/performance
    AI_PLAYER_KEY = "ai"
    HUMAN_PLAYER_KEY = "human"
    # Limit moves considered at root of minimax to save time, sorted by score first
    MAX_MOVES_TO_EVALUATE_AT_ROOT = 10

    @staticmethod
    def get_best_move(board_state: Board) -> Optional[Dict[str, Any]]:
        turn_start_time = time.time()
        # AI uses its internal raw rack for thinking
        ai_rack_raw = board_state.player_racks[AIPlayer.AI_PLAYER_KEY]
        ai_rack_display_for_log = "".join([POWER_TILE_TYPES.get(
            tile, {}).get("display", tile) for tile in ai_rack_raw])

        logger.info(
            f"AI ({AIPlayer.AI_PLAYER_KEY}) thinking. Rack (raw for AI): [{', '.join(ai_rack_raw)}] (Display: {ai_rack_display_for_log}). Limit: {AI_THINKING_TIME_LIMIT_SECONDS}s")

        best_move_info_overall = None  # Stores the dict for the chosen move
        best_heuristic_score_overall = -math.inf

        all_possible_moves_for_ai = board_state.get_valid_moves(
            AIPlayer.AI_PLAYER_KEY)

        if not all_possible_moves_for_ai:
            logger.info("AI found no valid moves. Will pass.")
            return None

        # Sort moves by their immediate score (descending) to explore promising ones first
        all_possible_moves_for_ai.sort(key=lambda m: m['score'], reverse=True)

        # Fallback: if minimax is too slow or depth is 0, pick highest scoring immediate move
        if all_possible_moves_for_ai:
            # Tentative best
            best_move_info_overall = all_possible_moves_for_ai[0]
            temp_board_for_baseline = deepcopy(board_state)

            # Manually apply effects of best_move_info_overall to temp_board_for_baseline
            # This needs to use the 'placed_info' which contains raw 'tile_marker'
            for tile_info_bs in best_move_info_overall['placed_info']:
                temp_board_for_baseline.board[tile_info_bs['pos'][0]][tile_info_bs['pos'][1]] = (
                    tile_info_bs['tile_marker'], True)

            temp_board_for_baseline.scores[AIPlayer.AI_PLAYER_KEY] += best_move_info_overall['base_score']
            if best_move_info_overall['objective_bonus'] > 0:
                temp_board_for_baseline.scores[AIPlayer.AI_PLAYER_KEY] += best_move_info_overall['objective_bonus']
                # Update objective in the *simulated* board state's objectives
                temp_board_for_baseline.player_objectives[AIPlayer.AI_PLAYER_KEY]['completed'] = True

            tiles_removed_bs = [info['tile_marker']
                                for info in best_move_info_overall['placed_info']]
            temp_board_for_baseline.remove_from_rack(
                tiles_removed_bs, AIPlayer.AI_PLAYER_KEY)
            temp_board_for_baseline.refill_rack(AIPlayer.AI_PLAYER_KEY)
            temp_board_for_baseline.first_move = False

            double_turn_triggered_bs = any(
                info['tile_marker'] == POWER_TILE_DOUBLE_TURN_MARKER for info in best_move_info_overall['placed_info'])
            if not double_turn_triggered_bs:
                temp_board_for_baseline.current_player = AIPlayer.HUMAN_PLAYER_KEY  # Switch turn for eval
            else:  # AI gets another turn
                temp_board_for_baseline.current_player = AIPlayer.AI_PLAYER_KEY

            best_heuristic_score_overall = AIPlayer.evaluate_board(
                temp_board_for_baseline)

        moves_for_minimax_eval = all_possible_moves_for_ai[:
                                                           AIPlayer.MAX_MOVES_TO_EVALUATE_AT_ROOT]
        logger.info(
            f"AI has {len(all_possible_moves_for_ai)} potential moves. Evaluating top {len(moves_for_minimax_eval)} with Minimax (Depth {AIPlayer.MAX_DEPTH}).")

        for idx, move_candidate_dict in enumerate(moves_for_minimax_eval):
            if time.time() - turn_start_time > AI_THINKING_TIME_LIMIT_SECONDS:
                logger.warning(
                    f"AI time limit reached after evaluating {idx} moves. Using best found so far.")
                break

            # Fresh copy for each candidate
            temp_board_sim = deepcopy(board_state)

            # Apply move_candidate_dict to temp_board_sim
            # Uses raw 'tile_marker'
            for tile_info_sim in move_candidate_dict['placed_info']:
                temp_board_sim.board[tile_info_sim['pos'][0]][tile_info_sim['pos'][1]] = (
                    tile_info_sim['tile_marker'], True)

            temp_board_sim.scores[AIPlayer.AI_PLAYER_KEY] += move_candidate_dict['base_score']
            if move_candidate_dict['objective_bonus'] > 0:
                temp_board_sim.scores[AIPlayer.AI_PLAYER_KEY] += move_candidate_dict['objective_bonus']
                # Update simulated objective
                temp_board_sim.player_objectives[AIPlayer.AI_PLAYER_KEY]['completed'] = True

            tiles_removed_sim = [info['tile_marker']
                                 for info in move_candidate_dict['placed_info']]
            temp_board_sim.remove_from_rack(
                tiles_removed_sim, AIPlayer.AI_PLAYER_KEY)
            temp_board_sim.refill_rack(AIPlayer.AI_PLAYER_KEY)
            temp_board_sim.first_move = False
            temp_board_sim.consecutive_passes = 0

            double_turn_sim = any(
                info['tile_marker'] == POWER_TILE_DOUBLE_TURN_MARKER for info in move_candidate_dict['placed_info'])

            next_minimax_is_maximizing = double_turn_sim
            if not double_turn_sim:
                temp_board_sim.current_player = AIPlayer.HUMAN_PLAYER_KEY
            else:
                temp_board_sim.current_player = AIPlayer.AI_PLAYER_KEY

            current_move_heuristic_eval = AIPlayer.minimax(
                temp_board_sim, AIPlayer.MAX_DEPTH - 1, -math.inf, math.inf, next_minimax_is_maximizing)

            if current_move_heuristic_eval > best_heuristic_score_overall:
                best_heuristic_score_overall = current_move_heuristic_eval
                best_move_info_overall = move_candidate_dict

        elapsed_time = time.time() - turn_start_time
        logger.info(f"AI evaluation loop finished in {elapsed_time:.3f}s.")

        if best_move_info_overall:
            logger.info(
                f"AI Chose: '{best_move_info_overall['word']}' (Segment of display chars) (Full Word Played: '{best_move_info_overall['full_word_played']}') at ({best_move_info_overall['row']},{best_move_info_overall['col']}) Dir:{best_move_info_overall['direction']}. Score:{best_move_info_overall['score']}. Est.Heuristic:{best_heuristic_score_overall:.2f}.")
            return {
                # This is the segment of display_chars for place_word
                "word": best_move_info_overall["word"],
                "row": best_move_info_overall["row"],
                "col": best_move_info_overall["col"],
                "direction": best_move_info_overall["direction"]
            }
        return None

    @staticmethod
    def minimax(board_state: Board, depth: int, alpha: float, beta: float, is_maximizing_player: bool) -> float:
        if depth == 0 or board_state.game_over:
            return AIPlayer.evaluate_board(board_state)

        current_player_for_moves_in_sim = AIPlayer.AI_PLAYER_KEY if is_maximizing_player else AIPlayer.HUMAN_PLAYER_KEY
        possible_moves_sim = board_state.get_valid_moves(
            current_player_for_moves_in_sim)

        if not possible_moves_sim:  # No moves, evaluate current board
            return AIPlayer.evaluate_board(board_state)

        if is_maximizing_player:  # AI's turn in simulation
            max_eval = -math.inf
            for move_sim_dict in possible_moves_sim:
                child_board_state = deepcopy(board_state)
                # Apply AI's simulated move (move_sim_dict) to child_board_state
                # Uses raw 'tile_marker'
                for tile_info_child in move_sim_dict['placed_info']:
                    child_board_state.board[tile_info_child['pos'][0]][tile_info_child['pos'][1]] = (
                        tile_info_child['tile_marker'], True)

                child_board_state.scores[AIPlayer.AI_PLAYER_KEY] += move_sim_dict['base_score']
                if move_sim_dict['objective_bonus'] > 0:
                    child_board_state.scores[AIPlayer.AI_PLAYER_KEY] += move_sim_dict['objective_bonus']
                    child_board_state.player_objectives[AIPlayer.AI_PLAYER_KEY]['completed'] = True

                tiles_removed_child = [info['tile_marker']
                                       for info in move_sim_dict['placed_info']]
                child_board_state.remove_from_rack(
                    tiles_removed_child, AIPlayer.AI_PLAYER_KEY)
                child_board_state.refill_rack(AIPlayer.AI_PLAYER_KEY)
                child_board_state.first_move = False
                child_board_state.consecutive_passes = 0

                double_turn_child_ai = any(
                    info['tile_marker'] == POWER_TILE_DOUBLE_TURN_MARKER for info in move_sim_dict['placed_info'])
                next_turn_is_maximizing = double_turn_child_ai
                if not double_turn_child_ai:
                    child_board_state.current_player = AIPlayer.HUMAN_PLAYER_KEY
                else:
                    child_board_state.current_player = AIPlayer.AI_PLAYER_KEY

                eval_score = AIPlayer.minimax(
                    child_board_state, depth - 1, alpha, beta, next_turn_is_maximizing)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:  # Human's turn in simulation (minimizing for AI)
            min_eval = math.inf
            for move_sim_dict in possible_moves_sim:
                child_board_state = deepcopy(board_state)
                # Apply Human's simulated move (move_sim_dict) to child_board_state
                # Uses raw 'tile_marker'
                for tile_info_child_h in move_sim_dict['placed_info']:
                    child_board_state.board[tile_info_child_h['pos'][0]][tile_info_child_h['pos'][1]] = (
                        tile_info_child_h['tile_marker'], True)

                child_board_state.scores[AIPlayer.HUMAN_PLAYER_KEY] += move_sim_dict['base_score']
                if move_sim_dict['objective_bonus'] > 0:
                    child_board_state.scores[AIPlayer.HUMAN_PLAYER_KEY] += move_sim_dict['objective_bonus']
                    child_board_state.player_objectives[AIPlayer.HUMAN_PLAYER_KEY]['completed'] = True

                tiles_removed_child_h = [info['tile_marker']
                                         for info in move_sim_dict['placed_info']]
                child_board_state.remove_from_rack(
                    tiles_removed_child_h, AIPlayer.HUMAN_PLAYER_KEY)
                child_board_state.refill_rack(AIPlayer.HUMAN_PLAYER_KEY)
                child_board_state.first_move = False
                child_board_state.consecutive_passes = 0

                double_turn_child_h = any(
                    info['tile_marker'] == POWER_TILE_DOUBLE_TURN_MARKER for info in move_sim_dict['placed_info'])
                next_turn_is_maximizing = not double_turn_child_h
                if not double_turn_child_h:
                    child_board_state.current_player = AIPlayer.AI_PLAYER_KEY
                else:
                    child_board_state.current_player = AIPlayer.HUMAN_PLAYER_KEY

                eval_score = AIPlayer.minimax(
                    child_board_state, depth - 1, alpha, beta, next_turn_is_maximizing)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval

    @staticmethod
    def evaluate_board(board_state: Board) -> float:
        """Heuristic evaluation of the board state from AI's perspective."""
        ai_score = board_state.scores.get(AIPlayer.AI_PLAYER_KEY, 0)
        human_score = board_state.scores.get(AIPlayer.HUMAN_PLAYER_KEY, 0)

        score_difference = ai_score - human_score

        objective_value = 0
        # Use the player_objectives from the simulated board_state
        if board_state.player_objectives[AIPlayer.AI_PLAYER_KEY]["completed"]:
            objective_value += 50
        if board_state.player_objectives[AIPlayer.HUMAN_PLAYER_KEY]["completed"]:
            objective_value -= 50

        rack_quality_score = 0
        # Raw rack from sim state
        ai_rack = board_state.player_racks[AIPlayer.AI_PLAYER_KEY]
        vowels = "AEIOU"
        num_vowels_in_rack = sum(1 for tile_marker in ai_rack if POWER_TILE_TYPES.get(
            tile_marker, {}).get("display", tile_marker) in vowels)
        num_consonants_in_rack = len(ai_rack) - num_vowels_in_rack
        rack_quality_score -= abs(num_vowels_in_rack -
                                  num_consonants_in_rack) * 1.5
        if ' ' in ai_rack:
            rack_quality_score += 10
        if 'S' in ai_rack:
            rack_quality_score += 5

        return score_difference + objective_value + rack_quality_score


def find_anchor_squares(board: Board) -> List[Tuple[int, int]]:
    """Finds squares where new words can be attached."""
    center_square = (board.size // 2, board.size // 2)
    if board.first_move:
        return [center_square]

    anchor_squares_set = set()
    for r in range(board.size):
        for c in range(board.size):
            if board.get_raw_tile_marker_at(r, c) is None:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    adj_r, adj_c = r + dr, c + dc
                    if 0 <= adj_r < board.size and 0 <= adj_c < board.size and \
                       board.get_raw_tile_marker_at(adj_r, adj_c) is not None:
                        anchor_squares_set.add((r, c))
                        break

    return list(anchor_squares_set) if anchor_squares_set else [center_square]


def generate_potential_placements_anchored(board: Board, player: str, anchor_squares: List[Tuple[int, int]]) -> List[Dict[str, Any]]:
    """
    Generates potential word placements (segments from rack) anchored to existing tiles.
    Returns list of dicts: {"word": "CAT", "row": R, "col": C, "direction": "horizontal/vertical"}
    "word" is the segment of *display characters* corresponding to tiles from the player's rack.
    """
    potential_placements_list = []
    # Get the raw rack ('A', ' ', 'D*')
    rack_raw_markers = board.player_racks[player]

    if not anchor_squares or not rack_raw_markers:
        return []

    current_max_permutation_len_to_check = min(
        len(rack_raw_markers), MAX_RACK_PERMUTATION_LENGTH)
    checked_placement_keys = set()

    for length_of_perm in range(1, current_max_permutation_len_to_check + 1):
        for p_tuple_raw in set(permutations(rack_raw_markers, length_of_perm)):
            # Convert tuple of raw markers to string of display characters for validate_move
            perm_str_display_chars = "".join([POWER_TILE_TYPES.get(tile_marker, {}).get(
                "display", tile_marker) for tile_marker in p_tuple_raw])

            for r_anchor, c_anchor in anchor_squares:
                for i_in_perm in range(length_of_perm):
                    start_col_h = c_anchor - i_in_perm
                    start_row_h = r_anchor
                    if 0 <= start_col_h and (start_col_h + length_of_perm - 1) < board.size:
                        placement_key_h = (
                            perm_str_display_chars, start_row_h, start_col_h, "horizontal")
                        if placement_key_h not in checked_placement_keys:
                            potential_placements_list.append({
                                "word": perm_str_display_chars,
                                "row": start_row_h, "col": start_col_h, "direction": "horizontal"
                            })
                            checked_placement_keys.add(placement_key_h)

                    start_row_v = r_anchor - i_in_perm
                    start_col_v = c_anchor
                    if 0 <= start_row_v and (start_row_v + length_of_perm - 1) < board.size:
                        placement_key_v = (
                            perm_str_display_chars, start_row_v, start_col_v, "vertical")
                        if placement_key_v not in checked_placement_keys:
                            potential_placements_list.append({
                                "word": perm_str_display_chars,
                                "row": start_row_v, "col": start_col_v, "direction": "vertical"
                            })
                            checked_placement_keys.add(placement_key_v)

    return potential_placements_list


class MoveRequest(BaseModel):
    # The letters the player wants to place (can include display char for power tile)
    word: str
    row: int
    col: int
    direction: str


class GameStateResponse(BaseModel):
    board: List[List[Optional[str]]]  # Display letters
    scores: Dict[str, int]
    current_player: str
    player_rack: List[str]  # Human's rack (display chars)
    game_over: bool
    message: Optional[str] = None
    first_move: bool
    tiles_in_bag: int
    human_objective: Optional[Dict[str, Any]] = None


game_board = Board()  # Global game board instance


@app.get("/api/game/start", response_model=GameStateResponse, tags=["Game Flow"])
async def start_game():
    global game_board
    game_board = Board()
    state = game_board.get_state()
    return GameStateResponse(
        board=state["board"],
        scores=state["scores"],
        current_player=state["current_player"],
        player_rack=state["player_rack"],
        game_over=state["game_over"],
        message="New game started. Your turn!",
        first_move=state["first_move"],
        tiles_in_bag=state["tiles_in_bag"],
        human_objective=state["human_objective"]
    )


@app.post("/api/game/move", response_model=GameStateResponse, tags=["Game Actions"])
async def player_move(move: MoveRequest):
    global game_board
    player_making_move = "human"

    if game_board.game_over:
        raise HTTPException(status_code=400, detail="Game is over.")
    if game_board.current_player != player_making_move:
        raise HTTPException(status_code=400, detail="Not your turn.")

    success, human_action_msg, _ = game_board.place_word(
        move.word, move.row, move.col, move.direction, player_making_move)

    if not success:
        state_after_invalid_human_move = game_board.get_state()
        return GameStateResponse(
            board=state_after_invalid_human_move["board"],
            scores=state_after_invalid_human_move["scores"],
            current_player=state_after_invalid_human_move["current_player"],
            player_rack=state_after_invalid_human_move["player_rack"],
            game_over=state_after_invalid_human_move["game_over"],
            # Prepend with a clear indicator
            message=f"Invalid Move: {human_action_msg}",
            first_move=state_after_invalid_human_move["first_move"],
            tiles_in_bag=state_after_invalid_human_move["tiles_in_bag"],
            human_objective=state_after_invalid_human_move["human_objective"]
        )

    # If human move was successful, check if AI needs to play
    ai_action_msg = ""
    if not game_board.game_over and game_board.current_player == "ai":
        logger.info("AI turn starting after human move.")
        ai_move_details = AIPlayer.get_best_move(
            game_board)  # Pass current board state
        if ai_move_details:
            ai_success, msg_from_ai_place, _ = game_board.place_word(
                ai_move_details["word"], ai_move_details["row"],
                ai_move_details["col"], ai_move_details["direction"], "ai"
            )
            ai_action_msg = msg_from_ai_place if ai_success else f"AI Error ({msg_from_ai_place}). AI Passes."
            # If AI's chosen move somehow fails validation (should be rare)
            if not ai_success:
                game_board.pass_turn("ai")  # AI passes if its move fails
        else:  # AI found no move / chose to pass
            _, msg_from_ai_pass = game_board.pass_turn("ai")
            ai_action_msg = f"AI Passes. ({msg_from_ai_pass})"
            logger.info("AI chose to pass or found no moves.")

    combined_message = f"You: {human_action_msg}"
    if ai_action_msg:
        combined_message += f" || AI: {ai_action_msg}"

    if game_board.game_over and "GAME OVER" not in combined_message.upper():
        combined_message += f" || GAME OVER! Final Score -> You: {game_board.scores.get('human',0)}, AI: {game_board.scores.get('ai',0)}"

    final_state_after_turn = game_board.get_state()
    return GameStateResponse(
        board=final_state_after_turn["board"],
        scores=final_state_after_turn["scores"],
        current_player=final_state_after_turn["current_player"],
        player_rack=final_state_after_turn["player_rack"],
        game_over=final_state_after_turn["game_over"],
        message=combined_message,
        first_move=final_state_after_turn["first_move"],
        tiles_in_bag=final_state_after_turn["tiles_in_bag"],
        human_objective=final_state_after_turn["human_objective"]
    )


@app.post("/api/game/pass", response_model=GameStateResponse, tags=["Game Actions"])
async def player_pass():
    global game_board
    player_passing = "human"

    if game_board.game_over:
        raise HTTPException(status_code=400, detail="Game is over.")
    if game_board.current_player != player_passing:
        raise HTTPException(status_code=400, detail="Not your turn to pass.")

    _, human_pass_msg = game_board.pass_turn(player_passing)

    ai_action_msg = ""
    if not game_board.game_over and game_board.current_player == "ai":
        logger.info("AI turn starting after human pass.")
        ai_move_details = AIPlayer.get_best_move(game_board)
        if ai_move_details:
            ai_success, msg_from_ai_place, _ = game_board.place_word(
                ai_move_details["word"], ai_move_details["row"],
                ai_move_details["col"], ai_move_details["direction"], "ai"
            )
            ai_action_msg = msg_from_ai_place if ai_success else f"AI Error ({msg_from_ai_place}). AI Passes."
            if not ai_success:
                game_board.pass_turn("ai")
        else:
            _, msg_from_ai_pass = game_board.pass_turn("ai")
            ai_action_msg = f"AI Passes. ({msg_from_ai_pass})"
            logger.info("AI chose to pass or found no moves.")

    combined_message = f"You: {human_pass_msg}"
    if ai_action_msg:
        combined_message += f" || AI: {ai_action_msg}"

    if game_board.game_over and "GAME OVER" not in combined_message.upper():
        combined_message += f" || GAME OVER! Final Score -> You: {game_board.scores.get('human',0)}, AI: {game_board.scores.get('ai',0)}"

    final_state_after_pass_turn = game_board.get_state()
    return GameStateResponse(
        board=final_state_after_pass_turn["board"],
        scores=final_state_after_pass_turn["scores"],
        current_player=final_state_after_pass_turn["current_player"],
        player_rack=final_state_after_pass_turn["player_rack"],
        game_over=final_state_after_pass_turn["game_over"],
        message=combined_message,
        first_move=final_state_after_pass_turn["first_move"],
        tiles_in_bag=final_state_after_pass_turn["tiles_in_bag"],
        human_objective=final_state_after_pass_turn["human_objective"]
    )


@app.get("/api/game/state", response_model=GameStateResponse, tags=["Game Info"])
async def get_current_game_state():
    global game_board
    current_state = game_board.get_state()
    return GameStateResponse(
        board=current_state["board"],
        scores=current_state["scores"],
        current_player=current_state["current_player"],
        player_rack=current_state["player_rack"],
        game_over=current_state["game_over"],
        message="Current game state retrieved.",
        first_move=current_state["first_move"],
        tiles_in_bag=current_state["tiles_in_bag"],
        human_objective=current_state["human_objective"]
    )

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Advanced Scrabble backend server...")
    if not VALID_WORDS or len(VALID_WORDS) < 50:  # Basic check for dictionary
        logger.critical(
            "Word dictionary issue: Dictionary is too small or not loaded. Exiting.")
        exit(1)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

# --- END OF FILE main.py ---
