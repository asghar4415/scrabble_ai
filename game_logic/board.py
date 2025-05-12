import logging
import random
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from .constants import (LETTER_SCORES, OBJECTIVE_TYPES,
                        POWER_TILE_DOUBLE_TURN_MARKER, POWER_TILE_TYPES)
from .utils import is_valid_word  # Import from utils

logger = logging.getLogger(__name__)


class Board:
    """Represents the Scrabble game state, including the board, scores, tiles, players, and rules."""

    def __init__(self):
        """Initializes a new game board state."""
        self.size = 15
        self.board: List[List[Optional[Tuple[str, bool]]]] = [
            [None for _ in range(self.size)] for _ in range(self.size)]
        self.scores = {"human": 0, "ai": 0}
        self.tile_bag = self.initialize_tile_bag()
        self.player_racks = {"human": self.draw_tiles(
            7), "ai": self.draw_tiles(7)}
        self.current_player = "human"
        self.game_over = False
        self.first_move = True
        self.premium_squares = self.initialize_premium_squares()
        self.last_move_time = time.time()
        self.consecutive_passes = 0
        self.player_objectives = {
            "human": self.assign_objective(), "ai": self.assign_objective()}
        self.power_tile_effect_active = {"double_turn": False}

    def get_letter_at(self, r: int, c: int) -> Optional[str]:
        """Gets the display letter at a board position (e.g., 'D' for 'D*' power tile)."""
        if 0 <= r < self.size and 0 <= c < self.size and self.board[r][c]:
            tile_marker = self.board[r][c][0]
            return POWER_TILE_TYPES.get(tile_marker, {}).get("display", tile_marker)
        return None

    def get_raw_tile_marker_at(self, r: int, c: int) -> Optional[str]:
        """Gets the raw internal tile marker (e.g., 'A', ' ', 'D*') at a board position."""
        if 0 <= r < self.size and 0 <= c < self.size and self.board[r][c]:
            return self.board[r][c][0]
        return None

    def initialize_tile_bag(self) -> List[str]:
        """Creates and shuffles the initial tile bag with standard Scrabble distribution plus power tiles."""
        distribution = {
            'A': 9, 'B': 2, 'C': 2, 'D': 4, 'E': 12, 'F': 2, 'G': 3, 'H': 2, 'I': 9, 'J': 1, 'K': 1, 'L': 4, 'M': 2,
            'N': 6, 'O': 8, 'P': 2, 'Q': 1, 'R': 6, 'S': 4, 'T': 6, 'U': 4, 'V': 2, 'W': 2, 'X': 1, 'Y': 2, 'Z': 1, ' ': 2
        }
        power_tiles_to_add = {POWER_TILE_DOUBLE_TURN_MARKER: 1}

        tile_bag = [letter for letter, count in distribution.items()
                    for _ in range(count)]
        tile_bag.extend(pt_marker for pt_marker,
                        count in power_tiles_to_add.items() for _ in range(count))
        random.shuffle(tile_bag)
        return tile_bag

    def draw_tiles(self, count: int) -> List[str]:
        """Draws a specified number of tiles from the bag."""
        drawn = []
        bag = self.tile_bag
        for _ in range(count):
            if not bag:
                break
            drawn.append(bag.pop())
        return drawn

    def initialize_premium_squares(self) -> Dict[Tuple[int, int], str]:
        """Sets up the dictionary mapping board coordinates to premium square types (TW, DW, TL, DL)."""
        premiums = {}
        size = self.size
        center = size // 2

        tw_coords = [(0, 0), (0, 7), (7, 0)]
        dw_coords = [(r, r) for r in range(1, 5)] + \
            [(r, size - 1 - r) for r in range(1, 5)]
        tl_coords = [(1, 5), (1, 9), (5, 1), (5, 5), (5, 9), (5, 13)]
        dl_coords = [
            (0, 3), (0, 11), (2, 6), (2, 8), (3, 0), (3, 7), (3, 14),
            (6, 2), (6, 6), (6, 8), (6, 12), (7, 3), (7, 11)
        ]

        all_coords = set((r, c) for r in range(size) for c in range(size))

        def mirror_and_add(base_coords, p_type):
            for r_orig, c_orig in base_coords:
                mirrored_points = [
                    (r_orig, c_orig), (r_orig, size - 1 - c_orig),
                    (size - 1 - r_orig, c_orig), (size -
                                                  1 - r_orig, size - 1 - c_orig)
                ]
                for mr, mc in mirrored_points:
                    if (mr, mc) in all_coords:
                        premiums[(mr, mc)] = p_type

        mirror_and_add(tw_coords, 'TW')
        mirror_and_add(dw_coords, 'DW')
        mirror_and_add(tl_coords, 'TL')
        mirror_and_add(dl_coords, 'DL')

        premiums[(center, center)] = 'DW'
        return premiums

    def assign_objective(self) -> Dict[str, Any]:
        """Randomly assigns a secret objective to a player."""
        obj_template = random.choice(OBJECTIVE_TYPES)
        return {"id": obj_template["id"], "desc": obj_template["desc"], "bonus": obj_template["bonus"], "completed": False}

    def calculate_move_score(self, word: str, row: int, col: int, direction: str, placed_tiles_info: List[Dict[str, Any]]) -> Tuple[int, List[str]]:
        """
        Calculates the total score for a validated move.
        Considers the main word, crosswords, premium squares, and bingo bonus.
        """
        total_score = 0
        words_formed_this_move = []
        newly_placed_positions = {info['pos'] for info in placed_tiles_info}

        main_word_score = 0
        main_word_multiplier = 1
        for i, letter_char in enumerate(word.upper()):
            current_r = row + (i if direction == "vertical" else 0)
            current_c = col + (i if direction == "horizontal" else 0)
            current_pos = (current_r, current_c)
            square_type = self.premium_squares.get(current_pos)
            is_newly_placed_tile = current_pos in newly_placed_positions
            letter_value = LETTER_SCORES.get(letter_char, 0)
            letter_multiplier_for_score = 1

            if is_newly_placed_tile:
                tile_info = next(
                    (info for info in placed_tiles_info if info['pos'] == current_pos), None)
                if tile_info and (tile_info['is_blank'] or tile_info['tile_marker'] in POWER_TILE_TYPES):
                    letter_value = 0
                if square_type == 'DL':
                    letter_multiplier_for_score = 2
                elif square_type == 'TL':
                    letter_multiplier_for_score = 3
                if square_type == 'DW':
                    main_word_multiplier *= 2
                elif square_type == 'TW':
                    main_word_multiplier *= 3

            main_word_score += letter_value * letter_multiplier_for_score

        main_word_score *= main_word_multiplier
        total_score += main_word_score
        if word not in words_formed_this_move:
            words_formed_this_move.append(word)

        cross_direction = "vertical" if direction == "horizontal" else "horizontal"
        for placed_tile_info in placed_tiles_info:
            r_placed, c_placed = placed_tile_info['pos']
            placed_letter_char = placed_tile_info['letter']
            cross_word_parts = [placed_letter_char]
            cross_word_start_pos_r, cross_word_start_pos_c = r_placed, c_placed

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

                    if pos_cw == (r_placed, c_placed):
                        if placed_tile_info['is_blank'] or placed_tile_info['tile_marker'] in POWER_TILE_TYPES:
                            val_cw = 0
                        letter_mult_cw = 1
                        if sq_type_cw == 'DL':
                            letter_mult_cw = 2
                        elif sq_type_cw == 'TL':
                            letter_mult_cw = 3
                        current_cross_word_score += val_cw * letter_mult_cw
                        if sq_type_cw == 'DW':
                            current_cross_word_multiplier *= 2
                        elif sq_type_cw == 'TW':
                            current_cross_word_multiplier *= 3
                    else:
                        current_cross_word_score += val_cw
                total_score += current_cross_word_score * current_cross_word_multiplier

        if len(newly_placed_positions) == 7:
            total_score += 50
        return total_score, sorted(list(set(words_formed_this_move)))

    def _reconstruct_full_line(self, start_row: int, start_col: int, direction: str,
                               length_of_placed_segment: int,
                               temp_board_view: List[List[Optional[Tuple[str, bool]]]]) -> Tuple[str, int, int]:
        """
        Internal helper to find the full word formed along a line, including existing tiles.
        """
        parts = []
        actual_start_r, actual_start_c = start_row, start_col

        if direction == "horizontal":
            current_c = start_col
            while current_c > 0 and temp_board_view[start_row][current_c - 1]:
                current_c -= 1
            actual_start_c = current_c
            end_c = start_col + length_of_placed_segment - 1
            while current_c <= end_c or (current_c < self.size and temp_board_view[start_row][current_c]):
                if current_c >= self.size:
                    break
                tile_info = temp_board_view[start_row][current_c]
                if tile_info:
                    display_char = POWER_TILE_TYPES.get(
                        tile_info[0], {}).get("display", tile_info[0])
                    parts.append(display_char)
                else:
                    if current_c <= end_c:
                        logger.error(
                            f"Gap in H-line reconstruction at ({start_row},{current_c})")
                        return "INVALID_LINE", start_row, actual_start_c
                    else:
                        break
                current_c += 1
        else:
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
                            f"Gap in V-line reconstruction at ({current_r},{start_col})")
                        return "INVALID_LINE", actual_start_r, start_col
                    else:
                        break
                current_r += 1
        return "".join(parts), actual_start_r, actual_start_c

    def validate_move(self, word_proposal_segment: str,
                      row_segment_start: int, col_segment_start: int, direction: str, player: str
                      ) -> Tuple[bool, str, List[Dict[str, Any]], str, int, int]:
        """
        Validates a player's proposed move segment against game rules.
        """
        rack = self.player_racks[player]
        placed_tiles_info_proposals = []
        connected_to_existing = self.first_move
        center_square = (self.size // 2, self.size // 2)
        touches_center_square = False
        num_new_tiles_from_rack = 0
        word_proposal_segment_upper = word_proposal_segment.upper()
        segment_len = len(word_proposal_segment_upper)

        if not word_proposal_segment_upper or not all(c.isalnum() or c == ' ' for c in word_proposal_segment_upper):
            return False, "Word proposal invalid format.", [], "", 0, 0
        if segment_len == 0:
            return False, "Must place at least one tile.", [], "", 0, 0

        if direction == "horizontal":
            if not (0 <= row_segment_start < self.size and 0 <= col_segment_start < self.size and 0 <= col_segment_start + segment_len - 1 < self.size):
                return False, "Placement out of bounds.", [], "", 0, 0
        else:
            if not (0 <= col_segment_start < self.size and 0 <= row_segment_start < self.size and 0 <= row_segment_start + segment_len - 1 < self.size):
                return False, "Placement out of bounds.", [], "", 0, 0

        temp_board_for_line_check = deepcopy(self.board)
        current_rack_copy = list(rack)
        for i in range(segment_len):
            r_curr = row_segment_start + (i if direction == "vertical" else 0)
            c_curr = col_segment_start + \
                (i if direction == "horizontal" else 0)
            pos_curr = (r_curr, c_curr)
            proposed_display_char_for_pos = word_proposal_segment_upper[i]
            existing_tile_marker_on_board = self.get_raw_tile_marker_at(
                r_curr, c_curr)
            existing_display_char_on_board = self.get_letter_at(r_curr, c_curr)

            if existing_tile_marker_on_board:
                if existing_display_char_on_board != proposed_display_char_for_pos:
                    return False, f"Conflict at ({r_curr},{c_curr}).", [], "", 0, 0
                connected_to_existing = True
            else:
                num_new_tiles_from_rack += 1
                found_rack_tile_for_char = False
                tile_marker_from_rack = ''
                is_blank_used = False

                if proposed_display_char_for_pos.isalpha() and proposed_display_char_for_pos in current_rack_copy:
                    tile_marker_from_rack = proposed_display_char_for_pos
                    current_rack_copy.remove(proposed_display_char_for_pos)
                    found_rack_tile_for_char = True
                elif not found_rack_tile_for_char:
                    for pt_m, pt_d in POWER_TILE_TYPES.items():
                        if pt_d.get("display") == proposed_display_char_for_pos and pt_m in current_rack_copy:
                            tile_marker_from_rack = pt_m
                            current_rack_copy.remove(pt_m)
                            found_rack_tile_for_char = True
                            break
                if not found_rack_tile_for_char and proposed_display_char_for_pos.isalpha() and ' ' in current_rack_copy:
                    tile_marker_from_rack = ' '
                    current_rack_copy.remove(' ')
                    is_blank_used = True
                    found_rack_tile_for_char = True

                if not found_rack_tile_for_char:
                    return False, f"Tile '{proposed_display_char_for_pos}' not in rack.", [], "", 0, 0

                placed_tiles_info_proposals.append({
                    'letter': proposed_display_char_for_pos, 'pos': pos_curr,
                    'is_blank': is_blank_used, 'tile_marker': tile_marker_from_rack
                })
                temp_board_for_line_check[r_curr][c_curr] = (
                    proposed_display_char_for_pos, True)

                if not self.first_move and not connected_to_existing:
                    for dr_adj, dc_adj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr_adj, nc_adj = r_curr + dr_adj, c_curr + dc_adj
                        if 0 <= nr_adj < self.size and 0 <= nc_adj < self.size and self.get_raw_tile_marker_at(nr_adj, nc_adj):
                            connected_to_existing = True
                            break
            if pos_curr == center_square:
                touches_center_square = True

        if num_new_tiles_from_rack == 0:
            return False, "No new tiles placed.", [], "", 0, 0

        if self.first_move and not touches_center_square:
            return False, "First move must cross center.", [], "", 0, 0
        if not self.first_move and not connected_to_existing:
            return False, "Move must connect to existing tiles.", [], "", 0, 0

        full_main_word, full_word_start_r, full_word_start_c = self._reconstruct_full_line(
            row_segment_start, col_segment_start, direction, segment_len, temp_board_for_line_check
        )
        if full_main_word == "INVALID_LINE" or not is_valid_word(full_main_word):
            return False, f"Main word '{full_main_word}' is not valid.", [], "", 0, 0

        cross_dir_check = "vertical" if direction == "horizontal" else "horizontal"
        for new_tile_info in placed_tiles_info_proposals:
            r_p, c_p = new_tile_info['pos']
            cross_word_str, _, _ = self._reconstruct_full_line(
                r_p, c_p, cross_dir_check, 1, temp_board_for_line_check)
            if len(cross_word_str) >= 2 and not is_valid_word(cross_word_str):
                return False, f"Creates invalid crossword '{cross_word_str}'.", [], "", 0, 0

        return True, f"Move forming '{full_main_word}' is valid.", placed_tiles_info_proposals, full_main_word, full_word_start_r, full_word_start_c

    def place_word(self, word_segment_from_player: str, row_segment_start: int, col_segment_start: int, direction: str, player: str) -> Tuple[bool, str, Optional[int]]:
        """
        Attempts to place a word on the board after validation.
        Updates scores, board state, player racks, checks for objectives, handles power-ups,
        checks for game over conditions, and switches the turn if necessary.
        """
        is_valid, message, resolved_placed_tiles_info, full_main_word_formed, full_word_r, full_word_c = self.validate_move(
            word_segment_from_player, row_segment_start, col_segment_start, direction, player)

        if not is_valid:
            return False, message, None

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
        letters_to_remove_from_rack_actual = []
        power_tile_triggered_effect = None

        for tile_info in resolved_placed_tiles_info:
            r_place, c_place = tile_info['pos']
            actual_tile_marker = tile_info['tile_marker']
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

        if not self.tile_bag and not self.player_racks[player]:
            self.game_over = True
            self.finalize_scores()

        if not self.game_over:
            if self.power_tile_effect_active["double_turn"]:
                self.power_tile_effect_active["double_turn"] = False
            else:
                self.switch_turn()

        success_message_main = f"Played '{full_main_word_formed}' for {final_score_for_turn} pts (Base: {score}). Words: {', '.join(words_formed_details)}."
        if objective_msg_part:
            success_message_main += objective_msg_part
        if power_tile_triggered_effect:
            success_message_main += f" Triggered {power_tile_triggered_effect}!"

        return True, success_message_main, final_score_for_turn

    def _get_word_start(self, r_start_segment: int, c_start_segment: int, direction: str) -> Tuple[int, int]:
        """Helper to find the actual start coordinates of a word line given a point on it."""
        r, c = r_start_segment, c_start_segment
        if direction == "horizontal":
            while c > 0 and self.get_raw_tile_marker_at(r, c - 1):
                c -= 1
        else:
            while r > 0 and self.get_raw_tile_marker_at(r - 1, c):
                r -= 1
        return r, c

    def check_objective_completion(self, objective_id: str, current_turn_score: int, words_this_turn: List[str], placed_tiles_info: List[Dict[str, Any]]) -> bool:
        """Checks if the conditions for a specific objective ID were met by the current move."""
        if objective_id == "score_gt_30":
            return current_turn_score >= 30
        if objective_id == "use_q_z_x_j":
            return any(info['letter'] in {'Q', 'Z', 'X', 'J'} for info in placed_tiles_info)
        if objective_id == "form_7_letter":
            return any(len(word) >= 7 for word in words_this_turn)
        if objective_id == "use_corner":
            corners = {(0, 0), (0, self.size-1), (self.size-1, 0),
                       (self.size-1, self.size-1)}
            return any(info['pos'] in corners for info in placed_tiles_info)
        return False

    def pass_turn(self, player: str) -> Tuple[bool, str]:
        """Handles a player passing their turn. Checks for game over due to consecutive passes."""
        self.consecutive_passes += 1
        if self.consecutive_passes >= 4:
            self.game_over = True
            self.finalize_scores()
            return True, f"{player} passed. Game over (consecutive passes)."
        self.switch_turn()
        return True, f"{player} passed turn."

    def finalize_scores(self):
        """Adjusts final scores at game end based on remaining rack tiles."""
        player_who_emptied_rack = next(
            (p for p in ["human", "ai"] if not self.player_racks[p]), None)

        if player_who_emptied_rack:
            opponent_key = self.get_opponent(player_who_emptied_rack)
            opponent_rack_value = sum(LETTER_SCORES.get(
                tm, 0) for tm in self.player_racks[opponent_key] if tm != ' ' and tm not in POWER_TILE_TYPES)
            self.scores[player_who_emptied_rack] += opponent_rack_value
            self.scores[opponent_key] -= opponent_rack_value
        else:
            for p_final in ["human", "ai"]:
                player_rack_value = sum(LETTER_SCORES.get(
                    tm, 0) for tm in self.player_racks[p_final] if tm != ' ' and tm not in POWER_TILE_TYPES)
                self.scores[p_final] -= player_rack_value

    def remove_from_rack(self, tile_markers_to_remove: List[str], player: str):
        """Removes specified tile markers from a player's rack."""
        rack = self.player_racks[player]
        for tile_marker in tile_markers_to_remove:
            try:
                rack.remove(tile_marker)
            except ValueError:
                logger.error(
                    f"Attempted to remove non-existent '{tile_marker}' from {player}'s rack {rack}")

    def refill_rack(self, player: str):
        """Refills a player's rack to 7 tiles from the bag, if possible."""
        needed = 7 - len(self.player_racks[player])
        if needed > 0:
            drawn_tiles = self.draw_tiles(needed)
            self.player_racks[player].extend(drawn_tiles)

    def switch_turn(self):
        """Switches the current player."""
        self.current_player = "ai" if self.current_player == "human" else "human"

    def get_opponent(self, player: str) -> str:
        """Returns the opponent of the given player."""
        return "ai" if player == "human" else "human"

    def get_state(self) -> Dict[str, Any]:
        """
        Returns the current game state suitable for the frontend API response.
        """
        display_board = []
        for r in range(self.size):
            row_display = [self.get_letter_at(r, c) for c in range(self.size)]
            display_board.append(row_display)

        human_rack_display = [POWER_TILE_TYPES.get(tile, {}).get(
            "display", tile) for tile in self.player_racks["human"]]

        return {
            "board": display_board,
            "scores": self.scores.copy(),
            "current_player": self.current_player,
            "player_rack": human_rack_display,
            "game_over": self.game_over,
            "first_move": self.first_move,
            "tiles_in_bag": len(self.tile_bag),
            "human_objective": self.player_objectives.get("human"),
            "internal_racks": self.player_racks.copy(),
            "internal_player_objectives": self.player_objectives.copy()
        }
