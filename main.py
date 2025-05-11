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
}
POWER_TILE_DOUBLE_TURN_MARKER = 'D*'
POWER_TILE_TYPES = {
    POWER_TILE_DOUBLE_TURN_MARKER: {"effect": "double_turn", "display": "D"}
}
OBJECTIVE_TYPES = [
    {"id": "score_gt_30", "desc": "Score 30+ points in a single turn", "bonus": 20},
    {"id": "use_q_z_x_j", "desc": "Play a word using Q, Z, X, or J", "bonus": 15},
    {"id": "form_7_letter",
        "desc": "Form a 7-letter word (Bingo already gives 50)", "bonus": 25},
    {"id": "use_corner",
        "desc": "Play a tile on a corner square (0,0/0,14/14,0/14,14)", "bonus": 10},
]

# Global constants for AI performance tuning
MAX_RACK_PERMUTATION_LENGTH = 5
# Slightly shorter for potentially faster AI vs AI
AI_THINKING_TIME_LIMIT_SECONDS = 5.0
AI_MAX_MOVES_TO_EVAL_HUMAN_VS_AI = 10
# Potentially fewer for faster AI vs AI spectacle
AI_MAX_MOVES_TO_EVAL_AI_VS_AI = 7


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
        words = loaded_words if loaded_words else minimal_word_set
        logger.info(
            f"Loaded {len(words)} words from {dict_path_found if loaded_words else 'minimal set'}")
        return words
    except Exception as e:
        logger.error(f"Error reading dictionary: {e}. Using minimal word set.")
        return minimal_word_set


VALID_WORDS = initialize_dictionary()


def is_valid_word(word: str) -> bool:
    if not word or len(word) < 2 or not word.isalpha():
        return False
    return word.upper() in VALID_WORDS


class Board:
    def __init__(self, game_mode: str = "human_vs_ai", ai1_name: str = "AI Alpha", ai2_name: str = "AI Beta"):
        self.size = 15
        self.board: List[List[Optional[Tuple[str, bool]]]] = [
            [None for _ in range(self.size)] for _ in range(self.size)]
        self.game_mode = game_mode
        self.player_keys: List[str] = []
        self.player_display_names: Dict[str, str] = {}

        if self.game_mode == "human_vs_ai":
            self.player_keys = ["human", "ai"]
            self.player_display_names = {"human": "You", "ai": "AI Opponent"}
        elif self.game_mode == "ai_vs_ai":
            self.player_keys = ["ai1", "ai2"]
            self.player_display_names = {"ai1": ai1_name, "ai2": ai2_name}
        else:
            raise ValueError(f"Unsupported game mode: {game_mode}")

        self.scores: Dict[str, int] = {key: 0 for key in self.player_keys}
        self.tile_bag = self.initialize_tile_bag()
        self.player_racks: Dict[str, List[str]] = {
            key: self.draw_tiles(7) for key in self.player_keys}
        self.current_player: str = self.player_keys[0]
        self.game_over = False
        self.first_move = True
        self.premium_squares = self.initialize_premium_squares()
        self.last_move_time = time.time()
        self.consecutive_passes = 0
        self.player_objectives: Dict[str, Any] = {
            key: self.assign_objective() for key in self.player_keys}
        self.power_tile_effect_active = {"double_turn": False}
        logger.info(
            f"Board initialized for {game_mode}. Players: {self.player_display_names}. Current turn: {self.player_display_names.get(self.current_player)}")

    def get_letter_at(self, r: int, c: int) -> Optional[str]:
        # ... (no change)
        if 0 <= r < self.size and 0 <= c < self.size and self.board[r][c]:
            tile_marker = self.board[r][c][0]
            return POWER_TILE_TYPES.get(tile_marker, {}).get("display", tile_marker)
        return None

    def get_raw_tile_marker_at(self, r: int, c: int) -> Optional[str]:
        # ... (no change)
        if 0 <= r < self.size and 0 <= c < self.size and self.board[r][c]:
            return self.board[r][c][0]
        return None

    def initialize_tile_bag(self) -> List[str]:
        # ... (no change)
        distribution = {'A': 9, 'B': 2, 'C': 2, 'D': 4, 'E': 12, 'F': 2, 'G': 3, 'H': 2, 'I': 9, 'J': 1, 'K': 1, 'L': 4, 'M': 2,
                        'N': 6, 'O': 8, 'P': 2, 'Q': 1, 'R': 6, 'S': 4, 'T': 6, 'U': 4, 'V': 2, 'W': 2, 'X': 1, 'Y': 2, 'Z': 1, ' ': 2}
        power_tiles_to_add = {POWER_TILE_DOUBLE_TURN_MARKER: 1}
        tile_bag = [letter for letter, count in distribution.items()
                    for _ in range(count)]
        tile_bag.extend(pt_marker for pt_marker,
                        count in power_tiles_to_add.items() for _ in range(count))
        random.shuffle(tile_bag)
        return tile_bag

    def draw_tiles(self, count: int) -> List[str]:
        # ... (no change)
        drawn = []
        for _ in range(count):
            if not self.tile_bag:
                break
            drawn.append(self.tile_bag.pop())
        return drawn

    def initialize_premium_squares(self) -> Dict[Tuple[int, int], str]:
        # ... (no change)
        premiums = {}
        size = self.size
        center = size // 2
        tw_coords = [(0, 0), (0, 7), (7, 0)]
        dw_coords = [(r, r) for r in range(1, 5)] + \
            [(center, center)] + [(r, size-1-r) for r in range(1, 5)]
        tl_coords = [(1, 5), (1, 9), (5, 1), (5, 5), (5, 9), (5, 13)]
        dl_coords = [(0, 3), (0, 11), (2, 6), (2, 8), (3, 0), (3, 7),
                     (3, 14), (6, 2), (6, 6), (6, 8), (6, 12), (7, 3), (7, 11)]
        all_coords = set((r, c) for r in range(size) for c in range(size))
        def mirror(r, c): return [(r, c), (r, size-1-c),
                                  (size-1-r, c), (size-1-r, size-1-c)]
        for coords, p_type in [(tw_coords, 'TW'), (dw_coords, 'DW'), (tl_coords, 'TL'), (dl_coords, 'DL')]:
            for r_orig, c_orig in coords:
                for mr, mc in mirror(r_orig, c_orig):
                    if (mr, mc) in all_coords:
                        premiums[(mr, mc)] = p_type
        premiums[(center, center)] = 'DW'
        return premiums

    def assign_objective(self) -> Dict[str, Any]:
        # ... (no change)
        obj = random.choice(OBJECTIVE_TYPES)
        return {"id": obj["id"], "desc": obj["desc"], "bonus": obj["bonus"], "completed": False}

    def calculate_move_score(self, word: str, row: int, col: int, direction: str, placed_tiles_info: List[Dict[str, Any]]) -> Tuple[int, List[str]]:
        # ... (no change from your last version, it was already generic) ...
        total_score = 0
        words_formed = []
        main_word_score = 0
        main_word_multiplier = 1
        placed_positions = {info['pos'] for info in placed_tiles_info}
        for i, letter in enumerate(word.upper()):
            r_curr = row + (i if direction == "vertical" else 0)
            c_curr = col + (i if direction == "horizontal" else 0)
            curr_pos = (r_curr, c_curr)
            square_type = self.premium_squares.get(curr_pos)
            is_newly_placed = curr_pos in placed_positions
            letter_val = LETTER_SCORES.get(letter, 0)
            letter_mult = 1
            if is_newly_placed:
                tile_info = next(
                    (info for info in placed_tiles_info if info['pos'] == curr_pos), None)
                if tile_info and (tile_info['is_blank'] or tile_info['tile_marker'] in POWER_TILE_TYPES):
                    letter_val = 0
                if square_type == 'DL':
                    letter_mult = 2
                elif square_type == 'TL':
                    letter_mult = 3
                elif square_type == 'DW':
                    main_word_multiplier *= 2
                elif square_type == 'TW':
                    main_word_multiplier *= 3
            main_word_score += letter_val * letter_mult
        main_word_score *= main_word_multiplier
        total_score += main_word_score
        words_formed.append(word)
        cross_dir = "vertical" if direction == "horizontal" else "horizontal"
        for info in placed_tiles_info:
            r_p, c_p = info['pos']
            p_letter = info['letter']
            cross_list = [p_letter]
            cross_start_pos = (r_p, c_p)
            cr, cc = r_p, c_p
            while True:
                nr, nc = (
                    cr - 1, cc) if cross_dir == "vertical" else (cr, cc - 1)
                if not (0 <= nr < self.size and 0 <= nc < self.size):
                    break
                l_add = self.get_letter_at(nr, nc)
                if l_add:
                    cross_list.insert(0, l_add)
                    cross_start_pos = (nr, nc)
                    cr, cc = nr, nc
                else:
                    break
            cr, cc = r_p, c_p
            while True:
                nr, nc = (
                    cr + 1, cc) if cross_dir == "vertical" else (cr, cc + 1)
                if not (0 <= nr < self.size and 0 <= nc < self.size):
                    break
                l_add = self.get_letter_at(nr, nc)
                if l_add:
                    cross_list.append(l_add)
                    cr, cc = nr, nc
                else:
                    break
            cross_w = "".join(cross_list)
            if len(cross_w) >= 2 and is_valid_word(cross_w) and cross_w not in words_formed:
                words_formed.append(cross_w)
                cross_s = 0
                cross_w_mult = 1
                for i_cw, l_cw in enumerate(cross_w.upper()):
                    r_cw = cross_start_pos[0] + \
                        (i_cw if cross_dir == "vertical" else 0)
                    c_cw = cross_start_pos[1] + \
                        (i_cw if cross_dir == "horizontal" else 0)
                    curr_cw_pos = (r_cw, c_cw)
                    sq_type_cw = self.premium_squares.get(curr_cw_pos)
                    l_val_cw = LETTER_SCORES.get(l_cw, 0)
                    tile_info_cw = next(
                        (pinf for pinf in placed_tiles_info if pinf['pos'] == curr_cw_pos), None)
                    if tile_info_cw and (tile_info_cw['is_blank'] or tile_info_cw['tile_marker'] in POWER_TILE_TYPES):
                        l_val_cw = 0
                    l_mult_cw = 1
                    if curr_cw_pos == (r_p, c_p):
                        if sq_type_cw == 'DL':
                            l_mult_cw = 2
                        elif sq_type_cw == 'TL':
                            l_mult_cw = 3
                        if sq_type_cw == 'DW':
                            cross_w_mult *= 2
                        elif sq_type_cw == 'TW':
                            cross_w_mult *= 3
                    cross_s += l_val_cw * l_mult_cw
                total_score += cross_s * cross_w_mult
        if len(placed_positions) == 7:
            total_score += 50
        return total_score, sorted(list(set(words_formed)))

    def _reconstruct_full_line(self, start_row: int, start_col: int, direction: str, length: int, temp_board_view: List[List[Optional[Tuple[str, bool]]]]) -> str:
        # ... (no change from your last version) ...
        parts = []
        if direction == "horizontal":
            true_start_c = start_col
            while true_start_c > 0 and temp_board_view[start_row][true_start_c - 1]:
                true_start_c -= 1
            true_end_c = start_col + length - 1
            while true_end_c < self.size - 1 and temp_board_view[start_row][true_end_c + 1]:
                true_end_c += 1
            for c_idx in range(true_start_c, true_end_c + 1):
                tile_info = temp_board_view[start_row][c_idx]
                if tile_info:
                    parts.append(tile_info[0])
                else:
                    logger.error(
                        f"Gap in horizontal line at ({start_row},{c_idx})")
                    return "INVALID_LINE"
        else:
            true_start_r = start_row
            while true_start_r > 0 and temp_board_view[true_start_r - 1][start_col]:
                true_start_r -= 1
            true_end_r = start_row + length - 1
            while true_end_r < self.size - 1 and temp_board_view[true_end_r + 1][start_col]:
                true_end_r += 1
            for r_idx in range(true_start_r, true_end_r + 1):
                tile_info = temp_board_view[r_idx][start_col]
                if tile_info:
                    parts.append(tile_info[0])
                else:
                    logger.error(
                        f"Gap in vertical line at ({r_idx},{start_col})")
                    return "INVALID_LINE"
        return "".join(parts)

    def validate_move(self, word_proposal: str, row: int, col: int, direction: str, player: str) -> Tuple[bool, str, List[Dict[str, Any]]]:
        # Uses the _reconstruct_full_line method now.
        # This method was substantially updated in the previous response and should be correct.
        # ... (ensure you have the full corrected version from the last AI "VANINFO" fix) ...
        rack = self.player_racks[player]
        placed_tiles_info = []
        letters_needed_from_rack = []
        connected_to_existing = self.first_move
        center_square = (self.size // 2, self.size // 2)
        touches_center = False
        num_new_tiles = 0
        word_proposal_upper = word_proposal.upper()

        if not word_proposal or not all(c.isalnum() or c == ' ' for c in word_proposal_upper):
            return False, "Word invalid.", []
        word_len = len(word_proposal_upper)

        if direction == "horizontal":
            if col < 0 or col + word_len > self.size or row < 0 or row >= self.size:
                return False, "Bounds.", []
        else:
            if row < 0 or row + word_len > self.size or col < 0 or col >= self.size:
                return False, "Bounds.", []

        # Tentatively place proposed tiles and gather info
        # Board to check full line validity
        temp_board_for_line_check = deepcopy(self.board)

        for i, proposed_char_from_rack_segment in enumerate(word_proposal_upper):
            r_current = row + (i if direction == "vertical" else 0)
            c_current = col + (i if direction == "horizontal" else 0)
            pos = (r_current, c_current)
            existing_tile_marker_on_main_board = self.get_raw_tile_marker_at(
                r_current, c_current)

            if existing_tile_marker_on_main_board:
                existing_display_letter = POWER_TILE_TYPES.get(existing_tile_marker_on_main_board, {
                }).get("display", existing_tile_marker_on_main_board)
                if existing_display_letter != proposed_char_from_rack_segment:
                    return False, f"Conflict: Board {existing_display_letter}, Propose {proposed_char_from_rack_segment}", []
                temp_board_for_line_check[r_current][c_current] = (
                    existing_display_letter, False)  # Mark as existing
                connected_to_existing = True
            else:
                num_new_tiles += 1
                letters_needed_from_rack.append(
                    proposed_char_from_rack_segment)
                placed_tiles_info.append(
                    {'letter': proposed_char_from_rack_segment, 'pos': pos, 'is_blank': False, 'tile_marker': '?'})
                temp_board_for_line_check[r_current][c_current] = (
                    proposed_char_from_rack_segment, True)  # Mark as new for line check
                if not connected_to_existing:
                    for dr_adj, dc_adj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr_adj, nc_adj = r_current+dr_adj, c_current+dc_adj
                        if 0 <= nr_adj < self.size and 0 <= nc_adj < self.size and self.get_raw_tile_marker_at(nr_adj, nc_adj):
                            connected_to_existing = True
                            break
            if self.first_move and pos == center_square:
                touches_center = True

        if num_new_tiles == 0:
            return False, "No new tiles.", []

        rack_copy = list(rack)
        final_resolved_placed_tiles_info = []
        possible_rack_use = True
        temp_placed_info_for_rack_check = list(
            placed_tiles_info)  # Use a copy for this loop

        for proposal_info in temp_placed_info_for_rack_check:
            needed_display_char = proposal_info['letter']
            original_pos = proposal_info['pos']
            found_rack_tile = False
            if needed_display_char.isalpha() and needed_display_char in rack_copy:
                rack_copy.remove(needed_display_char)
                final_resolved_placed_tiles_info.append(
                    {'letter': needed_display_char, 'pos': original_pos, 'is_blank': False, 'tile_marker': needed_display_char})
                found_rack_tile = True
            elif not found_rack_tile:
                for pt_marker, pt_data in POWER_TILE_TYPES.items():
                    if pt_data.get("display") == needed_display_char and pt_marker in rack_copy:
                        rack_copy.remove(pt_marker)
                        final_resolved_placed_tiles_info.append(
                            {'letter': needed_display_char, 'pos': original_pos, 'is_blank': False, 'tile_marker': pt_marker})
                        found_rack_tile = True
                        break
            if not found_rack_tile and needed_display_char.isalpha() and ' ' in rack_copy:
                rack_copy.remove(' ')
                final_resolved_placed_tiles_info.append(
                    {'letter': needed_display_char, 'pos': original_pos, 'is_blank': True, 'tile_marker': ' '})
                found_rack_tile = True
            if not found_rack_tile:
                possible_rack_use = False
                break

        if not possible_rack_use:
            return False, f"'{needed_display_char}' not in rack.", []
        placed_tiles_info = final_resolved_placed_tiles_info  # Use resolved info

        if self.first_move and not touches_center:
            return False, "First move center.", []
        if not self.first_move and not connected_to_existing:
            return False, "Must connect.", []

        # For main word validation, use the temp_board_for_line_check which has proposed tiles placed
        full_main_line_word = self._reconstruct_full_line(
            row, col, direction, word_len, temp_board_for_line_check)
        if full_main_line_word == "INVALID_LINE" or not is_valid_word(full_main_line_word):
            return False, f"Main line '{full_main_line_word}' invalid.", []

        cross_dir_check = "vertical" if direction == "horizontal" else "horizontal"
        for new_tile_info in placed_tiles_info:
            r_placed, c_placed = new_tile_info['pos']
            # For crossword check, use the temp_board_for_line_check which reflects the current proposal
            cross_word_formed = self._reconstruct_full_line(
                r_placed, c_placed, cross_dir_check, 1, temp_board_for_line_check)
            if len(cross_word_formed) >= 2 and not is_valid_word(cross_word_formed):
                return False, f"Crossword '{cross_word_formed}' invalid.", []

        return True, f"Move forming '{full_main_line_word}' valid.", placed_tiles_info

    def place_word(self, word: str, row: int, col: int, direction: str, player: str) -> Tuple[bool, str, Optional[int]]:
        # ... (no change from your last version, assuming it correctly uses `player` key) ...
        # This method relies on validate_move and then applies the changes.
        # The key is that `player_objectives[player]`, `scores[player]`, `player_racks[player]` are used.
        is_valid, message, placed_tiles_info = self.validate_move(
            word, row, col, direction, player)
        if not is_valid:
            return False, message, None

        # _get_formed_word_from_placement should use the *actual current board* + placed_tiles_info
        # to determine the word for scoring message, as validate_move already confirmed validity.
        # Or, we can use the full_main_line_word from validate_move's return if it was part of it.
        # For simplicity, let's use what was validated if possible, or reconstruct.
        # The `word` parameter to calculate_move_score should be the *full primary word formed*.

        # Reconstruct the primary word based on the *actual* placement for scoring
        # `row`, `col` here are start of the *placed segment*.
        # We need the start of the *full word* for calculate_move_score.
        full_word_start_r, full_word_start_c = self._get_word_start(
            row, col, direction)  # Start of the entire line
        # Reconstruct the word from this true start using the newly placed tiles
        word_for_scoring = self._get_formed_word_from_placement(
            full_word_start_r, full_word_start_c, direction, placed_tiles_info)

        score, words_formed = self.calculate_move_score(
            word_for_scoring, full_word_start_r, full_word_start_c, direction, placed_tiles_info)

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
            self.board[r_place][c_place] = (
                tile_marker, True)  # Store actual marker
            letters_removed_from_rack.append(tile_marker)
            if tile_marker in POWER_TILE_TYPES and POWER_TILE_TYPES[tile_marker].get("effect") == "double_turn":
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
                logger.info(
                    f"Player {self.player_display_names.get(player, player)} keeps turn due to Double Turn tile.")
                self.power_tile_effect_active["double_turn"] = False
            else:
                self.switch_turn()

        success_message = f"Played '{word_for_scoring}' for {score} pts (Base: {score-objective_bonus}). Words: {', '.join(words_formed)}."
        if objective_msg:
            success_message += objective_msg
        if power_tile_triggered:
            success_message += f" Triggered {power_tile_triggered}!"
        return True, success_message, score

    def _get_formed_word_from_placement(self, row: int, col: int, direction: str, placed_info: List[Dict[str, Any]]) -> str:
        # This function is called *after* validation to get the primary word string for scoring/messaging.
        # `row`, `col` should be the true start of the full word on the board.
        # `placed_info` contains only the newly placed tiles.
        parts = []
        current_r, current_c = row, col

        if direction == "horizontal":
            while current_c < self.size:
                existing_letter = self.get_letter_at(current_r, current_c)
                placed_tile_here = next(
                    (p_info['letter'] for p_info in placed_info if p_info['pos'] == (current_r, current_c)), None)

                if placed_tile_here:
                    parts.append(placed_tile_here)
                elif existing_letter:
                    parts.append(existing_letter)
                else:
                    break  # End of word
                current_c += 1
        else:  # Vertical
            while current_r < self.size:
                existing_letter = self.get_letter_at(current_r, current_c)
                placed_tile_here = next(
                    (p_info['letter'] for p_info in placed_info if p_info['pos'] == (current_r, current_c)), None)

                if placed_tile_here:
                    parts.append(placed_tile_here)
                elif existing_letter:
                    parts.append(existing_letter)
                else:
                    break
                current_r += 1
        return "".join(parts)

    def _get_word_start(self, r_place_segment: int, c_place_segment: int, direction: str) -> Tuple[int, int]:
        # Given the start of the *placed segment*, find the true start of the *entire word line*.
        start_r, start_c = r_place_segment, c_place_segment
        if direction == "horizontal":
            while start_c > 0 and self.get_raw_tile_marker_at(start_r, start_c - 1):
                start_c -= 1
        else:
            while start_r > 0 and self.get_raw_tile_marker_at(start_r - 1, start_c):
                start_r -= 1
        return start_r, start_c

    def check_objective_completion(self, objective_id: str, move_score: int, words_formed: List[str], placed_tiles_info: List[Dict[str, Any]]) -> bool:
        # ... (no change)
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
        # ... (no change)
        if player != self.current_player:
            return False, "Not your turn."
        self.consecutive_passes += 1
        logger.info(
            f"Player {self.player_display_names.get(player,player)} passed. Consecutive: {self.consecutive_passes}")
        if self.consecutive_passes >= 6:
            self.game_over = True
            self.finalize_scores()
            return True, "Turn passed. Game Over: 6 consecutive passes."
        else:
            self.switch_turn()
            return True, "Turn passed."

    def finalize_scores(self):
        # ... (no change, uses self.player_keys)
        logger.info(f"Game Over ({self.game_mode}) - Finalizing Scores.")
        empty_rack_player_key = None
        total_unplayed_score = 0
        for p_key in self.player_keys:
            rack = self.player_racks[p_key]
            p_unplayed_score = sum(LETTER_SCORES.get(POWER_TILE_TYPES.get(
                tile, {}).get("display", tile), 0) for tile in rack)
            if not rack:
                empty_rack_player_key = p_key
            else:
                self.scores[p_key] -= p_unplayed_score
                total_unplayed_score += p_unplayed_score
        if empty_rack_player_key:
            opponent_key = self.get_opponent(empty_rack_player_key)
            if opponent_key in self.scores:
                self.scores[empty_rack_player_key] += total_unplayed_score
        final_score_log = "Final Scores: "
        for p_key in self.player_keys:
            final_score_log += f"{self.player_display_names.get(p_key, p_key)}: {self.scores[p_key]} "
        logger.info(final_score_log)

    def remove_from_rack(self, tile_markers: List[str], player: str):
        # ... (no change)
        rack = self.player_racks[player]
        for marker in tile_markers:
            try:
                rack.remove(marker)
            except ValueError:
                logger.error(
                    f"CRITICAL: Failed to remove '{marker}' from {player}'s rack {rack}")

    def refill_rack(self, player: str):
        # ... (no change)
        needed = 7 - len(self.player_racks[player])
        if needed > 0:
            new_tiles = self.draw_tiles(needed)
            if new_tiles:
                self.player_racks[player].extend(new_tiles)
        if not self.tile_bag:
            logger.info("Tile bag is empty.")

    # get_opponent, switch_turn, get_state, get_valid_moves are now correctly using player keys

    @staticmethod  # Helper for AI simulation
    def apply_simulated_move_for_player(board_instance: 'Board', move_dict: Dict, player_key: str):
        # ... (This static method was correctly defined previously)
        letters_removed = [info['tile_marker']
                           for info in move_dict['placed_info']]
        for info in move_dict['placed_info']:
            board_instance.board[info['pos'][0]][info['pos'][1]] = (
                info['tile_marker'], True)
        board_instance.scores[player_key] += move_dict['base_score']
        if move_dict['objective_bonus'] > 0:
            board_instance.scores[player_key] += move_dict['objective_bonus']
            board_instance.player_objectives[player_key]['completed'] = True
        board_instance.remove_from_rack(letters_removed, player_key)
        board_instance.refill_rack(player_key)
        is_double_turn = any(
            info['tile_marker'] == POWER_TILE_DOUBLE_TURN_MARKER for info in move_dict['placed_info'])
        if is_double_turn:
            board_instance.current_player = player_key
        else:
            board_instance.current_player = board_instance.get_opponent(
                player_key)
        board_instance.first_move = False
        board_instance.consecutive_passes = 0


# AIPlayer, find_anchor_squares, generate_potential_placements_anchored,
# API Models, and API Endpoints should largely be the same as the last complete version,
# ensuring that AIPlayer.get_best_move passes the correct `ai_player_key`
# and that API endpoints correctly instantiate Board with the right mode and names.
# I've included the full updated AIPlayer to be sure.

class AIPlayer:
    MAX_DEPTH = 1
    # MAX_MOVES_TO_EVALUATE_AT_ROOT is passed as arg now
    # AI_THINKING_TIME_LIMIT_SECONDS is global

    @staticmethod
    def get_best_move(board: Board, ai_player_key: str, max_moves_to_eval: int) -> Optional[Dict[str, Any]]:
        turn_start_time = time.time()
        ai_display_name = board.player_display_names.get(
            ai_player_key, ai_player_key)
        logger.info(
            f"AI ({ai_display_name}) starting. Rack: {board.player_racks[ai_player_key]}. Time limit: {AI_THINKING_TIME_LIMIT_SECONDS}s")

        best_move_overall = None
        best_heuristic_score_overall = -math.inf
        all_possible_moves = board.get_valid_moves(ai_player_key)
        if not all_possible_moves:
            logger.info(f"AI {ai_display_name} found no valid moves.")
            return None
        all_possible_moves.sort(key=lambda m: m['score'], reverse=True)

        if all_possible_moves:  # Initialize fallback
            best_move_overall = all_possible_moves[0]
            temp_board_fb = deepcopy(board)
            Board.apply_simulated_move_for_player(
                temp_board_fb, best_move_overall, ai_player_key)
            best_heuristic_score_overall = AIPlayer.evaluate_board(
                temp_board_fb, ai_player_key)

        moves_for_minimax_eval = all_possible_moves[:max_moves_to_eval]
        logger.info(
            f"AI {ai_display_name} has {len(all_possible_moves)} moves. Eval top {len(moves_for_minimax_eval)}.")

        for idx, current_move_candidate in enumerate(moves_for_minimax_eval):
            if time.time() - turn_start_time > AI_THINKING_TIME_LIMIT_SECONDS:
                logger.warning(
                    f"AI {ai_display_name} timed out. Returning best found.")
                break
            temp_board_for_sim = deepcopy(board)
            Board.apply_simulated_move_for_player(
                temp_board_for_sim, current_move_candidate, ai_player_key)

            is_double_turn = any(
                info['tile_marker'] == POWER_TILE_DOUBLE_TURN_MARKER for info in current_move_candidate['placed_info'])
            # If current_move_candidate results in original AI playing again, next state is maximizing for original AI
            # Otherwise, opponent plays, so next state is minimizing for original AI
            next_state_is_maximizing_for_original_ai = is_double_turn

            heuristic_eval = AIPlayer.minimax(temp_board_for_sim, AIPlayer.MAX_DEPTH - 1, -math.inf, math.inf,
                                              next_state_is_maximizing_for_original_ai, ai_player_key)
            if heuristic_eval > best_heuristic_score_overall:
                best_heuristic_score_overall = heuristic_eval
                best_move_overall = current_move_candidate

        if best_move_overall:
            logger.info(
                f"AI {ai_display_name} Chose: {best_move_overall['word']} (Heuristic: {best_heuristic_score_overall})")
            return {"word": best_move_overall["word"], "row": best_move_overall["row"],
                    "col": best_move_overall["col"], "direction": best_move_overall["direction"]}
        return None

    @staticmethod
    def minimax(board_state: Board, depth: int, alpha: float, beta: float,
                is_maximizing_for_original_ai: bool, original_ai_player_key: str) -> float:
        if depth == 0 or board_state.game_over:
            return AIPlayer.evaluate_board(board_state, original_ai_player_key)

        # The player whose turn it ACTUALLY IS in this board_state
        current_sim_player_key = board_state.current_player
        possible_moves = board_state.get_valid_moves(current_sim_player_key)
        if not possible_moves:
            return AIPlayer.evaluate_board(board_state, original_ai_player_key)

        # We evaluate based on whether the current_sim_player_key is the original_ai_player_key
        # If it IS the original_ai_player_key's turn, we want to maximize ITS score (from ITS perspective)
        # If it's the OPPONENT'S turn, we want to minimize the original_ai_player_key's score (by maximizing opponent's relative advantage)

        # This logic was slightly off. The `is_maximizing_for_original_ai` flag passed into minimax
        # should determine if we are in a max or min node *from the perspective of the original AI*.
        if is_maximizing_for_original_ai:  # We are trying to find a move that is good for original_ai_player_key
            max_eval = -math.inf
            for move in possible_moves:
                child_state = deepcopy(board_state)
                Board.apply_simulated_move_for_player(
                    child_state, move, current_sim_player_key)  # Apply move for current_sim_player

                # Determine if the *next* call to minimax is still maximizing for original_ai_player_key
                is_double_turn = any(
                    info['tile_marker'] == POWER_TILE_DOUBLE_TURN_MARKER for info in move['placed_info'])
                # current_sim_player (who is original_ai) goes again
                if is_double_turn:
                    next_call_is_maximizing_for_original = True
                else:  # Turn switches to opponent
                    next_call_is_maximizing_for_original = False

                eval_score = AIPlayer.minimax(
                    child_state, depth - 1, alpha, beta, next_call_is_maximizing_for_original, original_ai_player_key)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        # We are in a state where the opponent of original_ai_player_key is playing; we want to find their best move (which is worst for original_ai)
        else:
            min_eval = math.inf
            for move in possible_moves:
                child_state = deepcopy(board_state)
                # Apply move for current_sim_player (opponent)
                Board.apply_simulated_move_for_player(
                    child_state, move, current_sim_player_key)

                is_double_turn = any(
                    info['tile_marker'] == POWER_TILE_DOUBLE_TURN_MARKER for info in move['placed_info'])
                if is_double_turn:  # opponent goes again
                    next_call_is_maximizing_for_original = False
                else:  # Turn switches to original_ai_player_key
                    next_call_is_maximizing_for_original = True

                eval_score = AIPlayer.minimax(
                    child_state, depth - 1, alpha, beta, next_call_is_maximizing_for_original, original_ai_player_key)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval

    @staticmethod
    def evaluate_board(board_state: Board, for_player_key: str) -> float:
        # ... (no change from your last version)
        perspective_score = board_state.scores.get(for_player_key, 0)
        # Make sure get_opponent handles ai1/ai2
        opponent_key = board_state.get_opponent(for_player_key)
        opponent_score = board_state.scores.get(opponent_key, 0)
        score_diff = perspective_score - opponent_score
        obj_bonus = 50 if board_state.player_objectives[for_player_key]["completed"] else 0
        opp_obj_penalty = -50 if board_state.player_objectives.get(
            opponent_key, {"completed": False})["completed"] else 0  # Safer get
        rack_value = 0
        # ... (rest of rack eval)
        return score_diff + obj_bonus + opp_obj_penalty + rack_value


# --- API Models and Endpoints ---
# (These should be mostly the same as the last complete version provided,
#  just ensure the new global game_board: Optional[Board] = None is used,
#  and Board() is instantiated with the correct game_mode and names)

class StartAIGameRequest(BaseModel):
    ai1_name: str = "RoboScrabbler"
    ai2_name: str = "WordWizard"


class GameStateResponse(BaseModel):
    board: List[List[Optional[str]]]
    scores: Dict[str, int]
    current_player_key: str
    current_player_display_name: str
    player_racks_display: Dict[str, List[str]]
    player_display_names: Dict[str, str]
    game_over: bool
    message: Optional[str] = None
    first_move: bool
    tiles_in_bag: int
    game_mode: str
    player_objectives: Dict[str, Any]


class MoveRequest(BaseModel):
    word: str
    row: int
    col: int
    direction: str


game_board: Optional[Board] = None


@app.post("/api/game/start/human_vs_ai", response_model=GameStateResponse, tags=["Game Flow"])
async def start_human_vs_ai_game():
    global game_board
    game_board = Board(game_mode="human_vs_ai")
    logger.info("--- New Human vs AI Game Started ---")
    return GameStateResponse(**game_board.get_state())


@app.post("/api/game/start/ai_vs_ai", response_model=GameStateResponse, tags=["Game Flow"])
async def start_ai_vs_ai_game(names: StartAIGameRequest):
    global game_board
    game_board = Board(game_mode="ai_vs_ai",
                       ai1_name=names.ai1_name, ai2_name=names.ai2_name)
    logger.info(
        f"--- New AI vs AI Game: {names.ai1_name} vs {names.ai2_name} ---")
    return GameStateResponse(**game_board.get_state())


@app.post("/api/game/move", response_model=GameStateResponse, tags=["Game Actions"])
async def human_player_move(move: MoveRequest):
    global game_board
    if not game_board or game_board.game_mode != "human_vs_ai":
        raise HTTPException(status_code=400, detail="Not Human vs AI game.")
    player_key = "human"
    if game_board.current_player != player_key:
        raise HTTPException(status_code=403, detail="Not your turn.")
    if game_board.game_over:
        raise HTTPException(status_code=400, detail="Game over.")
    success, human_msg, _ = game_board.place_word(
        move.word, move.row, move.col, move.direction, player_key)
    if not success:
        return GameStateResponse(**game_board.get_state(), message=f"Invalid: {human_msg}")
    ai_msg_part = ""
    if not game_board.game_over and game_board.current_player == "ai":
        ai_move = AIPlayer.get_best_move(
            game_board, "ai", AI_MAX_MOVES_TO_EVAL_HUMAN_VS_AI)
        if ai_move:
            ai_success, msg_ai, _ = game_board.place_word(
                ai_move["word"], ai_move["row"], ai_move["col"], ai_move["direction"], "ai")
            ai_msg_part = msg_ai if ai_success else f"AI Error ({msg_ai}). AI Passes."
            if not ai_success:
                game_board.pass_turn("ai")
        else:
            _, pass_msg_ai = game_board.pass_turn("ai")
            ai_msg_part = f"AI passes. ({pass_msg_ai})"
    final_message = f"Your move: {human_msg}" + \
        (f" || AI's move: {ai_msg_part}" if ai_msg_part else "")
    # ... (game over append)
    return GameStateResponse(**game_board.get_state(), message=final_message)


@app.post("/api/game/pass", response_model=GameStateResponse, tags=["Game Actions"])
async def human_player_pass():
    global game_board
    if not game_board or game_board.game_mode != "human_vs_ai":
        raise HTTPException(status_code=400, detail="Not Human vs AI game.")
    player_key = "human"
    if game_board.current_player != player_key:
        raise HTTPException(status_code=403, detail="Not your turn.")
    if game_board.game_over:
        raise HTTPException(status_code=400, detail="Game over.")
    _, pass_msg = game_board.pass_turn(player_key)
    ai_msg_part = ""
    if not game_board.game_over and game_board.current_player == "ai":
        ai_move = AIPlayer.get_best_move(
            game_board, "ai", AI_MAX_MOVES_TO_EVAL_HUMAN_VS_AI)
        if ai_move:
            ai_success, msg_ai, _ = game_board.place_word(
                ai_move["word"], ai_move["row"], ai_move["col"], ai_move["direction"], "ai")
            ai_msg_part = msg_ai if ai_success else f"AI Error ({msg_ai}). AI Passes."
            if not ai_success:
                game_board.pass_turn("ai")
        else:
            _, pass_msg_ai = game_board.pass_turn("ai")
            ai_msg_part = f"AI passes. ({pass_msg_ai})"
    final_message = f"You passed: {pass_msg}" + \
        (f" || AI's move: {ai_msg_part}" if ai_msg_part else "")
    # ... (game over append)
    return GameStateResponse(**game_board.get_state(), message=final_message)


@app.post("/api/game/ai_vs_ai/next_move", response_model=GameStateResponse, tags=["AI vs AI"])
async def ai_vs_ai_next_move():
    global game_board
    if not game_board or game_board.game_mode != "ai_vs_ai":
        raise HTTPException(status_code=400, detail="Not AI vs AI game.")
    if game_board.game_over:
        raise HTTPException(status_code=400, detail="Game over.")
    current_ai_player_key = game_board.current_player
    if current_ai_player_key not in game_board.player_keys:
        # More generic check
        raise HTTPException(status_code=500, detail="Invalid player key.")

    ai_display_name = game_board.player_display_names.get(
        current_ai_player_key, current_ai_player_key)
    logger.info(f"AI vs AI: {ai_display_name}'s turn...")
    ai_move_action = AIPlayer.get_best_move(
        game_board, current_ai_player_key, AI_MAX_MOVES_TO_EVAL_AI_VS_AI)
    move_message = ""
    if ai_move_action:
        success, msg, _ = game_board.place_word(
            ai_move_action["word"], ai_move_action["row"], ai_move_action["col"], ai_move_action["direction"], current_ai_player_key)
        move_message = msg if success else f"AI {ai_display_name} Error ({msg}). Passing."
        if not success:
            game_board.pass_turn(current_ai_player_key)
    else:
        _, pass_msg = game_board.pass_turn(current_ai_player_key)
        move_message = f"AI {ai_display_name} passes. ({pass_msg})"
    final_message = f"{ai_display_name}'s move: {move_message}"
    # ... (game over append)
    return GameStateResponse(**game_board.get_state(), message=final_message)


@app.get("/api/game/state", response_model=GameStateResponse, tags=["Game Info"])
async def get_current_game_state():
    global game_board
    if not game_board:
        raise HTTPException(status_code=404, detail="Game not started.")
    return GameStateResponse(**game_board.get_state())

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Advanced Scrabble backend server...")
    if not VALID_WORDS or len(VALID_WORDS) < 50:
        logger.critical(f"Word dictionary issue. Exiting.")
        exit(1)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

# --- END OF FILE main.py ---
