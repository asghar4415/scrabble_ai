
import logging
import math
import time
from copy import deepcopy
from itertools import permutations
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from .constants import LETTER_SCORES

from .constants import (POWER_TILE_TYPES, POWER_TILE_DOUBLE_TURN_MARKER,
                        MAX_RACK_PERMUTATION_LENGTH, AI_THINKING_TIME_LIMIT_SECONDS)
from .utils import find_anchor_squares


if TYPE_CHECKING:
    from .board import Board  # Avoid circular import for type hinting

logger = logging.getLogger(__name__)


def generate_potential_placements_anchored(board: 'Board', player: str, anchor_squares: List[tuple[int, int]]) -> List[Dict[str, Any]]:
    placements = []
    rack_raw = board.player_racks[player]
    if not anchor_squares or not rack_raw:
        return []

    max_perm_len = min(len(rack_raw), MAX_RACK_PERMUTATION_LENGTH)
    checked_keys = set()

    for length in range(1, max_perm_len + 1):
        for p_tuple in set(permutations(rack_raw, length)):
            perm_display = "".join([POWER_TILE_TYPES.get(
                tm, {}).get("display", tm) for tm in p_tuple])
            for r_anchor, c_anchor in anchor_squares:
                for i in range(length):
                    start_col_h = c_anchor - i
                    start_row_h = r_anchor
                    if 0 <= start_col_h and (start_col_h + length - 1) < board.size:
                        key_h = (perm_display, start_row_h,
                                 start_col_h, "horizontal")
                        if key_h not in checked_keys:
                            placements.append(
                                {"word": perm_display, "row": start_row_h, "col": start_col_h, "direction": "horizontal"})
                            checked_keys.add(key_h)
                    start_row_v = r_anchor - i
                    start_col_v = c_anchor
                    if 0 <= start_row_v and (start_row_v + length - 1) < board.size:
                        key_v = (perm_display, start_row_v,
                                 start_col_v, "vertical")
                        if key_v not in checked_keys:
                            placements.append(
                                {"word": perm_display, "row": start_row_v, "col": start_col_v, "direction": "vertical"})
                            checked_keys.add(key_v)
    return placements


class AIPlayer:
    MAX_DEPTH = 1
    AI_PLAYER_KEY = "ai"
    HUMAN_PLAYER_KEY = "human"
    MAX_MOVES_TO_EVALUATE_AT_ROOT = 10

    @staticmethod
    def _get_valid_moves_for_player(board_instance: 'Board', player: str) -> List[Dict[str, Any]]:
        valid_moves = []
        anchor_sqs = find_anchor_squares(board_instance)
        potential_placements = generate_potential_placements_anchored(
            board_instance, player, anchor_sqs)

        for placement_input in potential_placements:
            word_segment_display_chars = placement_input["word"]
            row, col, direction = placement_input["row"], placement_input["col"], placement_input["direction"]

            is_valid, _, placed_info, full_word, full_r, full_c = board_instance.validate_move(
                word_segment_display_chars, row, col, direction, player)

            if is_valid:
                base_score, temp_words_formed = board_instance.calculate_move_score(
                    full_word, full_r, full_c, direction, placed_info)
                objective_bonus = 0
                player_obj = board_instance.player_objectives[player]
                if not player_obj["completed"] and board_instance.check_objective_completion(player_obj["id"], base_score, temp_words_formed, placed_info):
                    objective_bonus = player_obj["bonus"]

                valid_moves.append({
                    "word": word_segment_display_chars, "row": row, "col": col, "direction": direction,
                    "score": base_score + objective_bonus, "base_score": base_score,
                    "objective_bonus": objective_bonus, "placed_info": placed_info,
                    "full_word_played": full_word
                })
        return valid_moves

    @staticmethod
    def get_best_move(board_state: 'Board') -> Optional[Dict[str, Any]]:
        turn_start_time = time.time()
        ai_rack_raw = board_state.player_racks[AIPlayer.AI_PLAYER_KEY]
        ai_rack_display_log = "".join([POWER_TILE_TYPES.get(
            t, {}).get("display", t) for t in ai_rack_raw])
        logger.info(
            f"AI ({AIPlayer.AI_PLAYER_KEY}) thinking. Rack: [{', '.join(ai_rack_raw)}] (Display: {ai_rack_display_log}). Limit: {AI_THINKING_TIME_LIMIT_SECONDS}s")

        all_moves = AIPlayer._get_valid_moves_for_player(
            board_state, AIPlayer.AI_PLAYER_KEY)
        if not all_moves:
            logger.info("AI found no valid moves. Passing.")
            return None

        all_moves.sort(key=lambda m: m['score'], reverse=True)
        best_move_info = all_moves[0]
        best_heuristic_score = -math.inf

        temp_board_baseline = deepcopy(board_state)
        best_heuristic_score = AIPlayer.evaluate_board(
            temp_board_baseline)  # Initial placeholder

        moves_to_evaluate = all_moves[:AIPlayer.MAX_MOVES_TO_EVALUATE_AT_ROOT]
        logger.info(
            f"AI evaluating top {len(moves_to_evaluate)}/{len(all_moves)} moves with Minimax (Depth {AIPlayer.MAX_DEPTH}).")

        for idx, move_candidate in enumerate(moves_to_evaluate):
            if time.time() - turn_start_time > AI_THINKING_TIME_LIMIT_SECONDS:
                logger.warning(
                    f"AI time limit hit after {idx} evaluations. Using best found.")
                break

            temp_board_sim = deepcopy(board_state)

            double_turn_sim = any(
                info['tile_marker'] == POWER_TILE_DOUBLE_TURN_MARKER for info in move_candidate['placed_info'])
            next_is_maximizing = double_turn_sim

            for tile_info_sim in move_candidate['placed_info']:  # Raw markers
                temp_board_sim.board[tile_info_sim['pos'][0]][tile_info_sim['pos'][1]] = (
                    tile_info_sim['tile_marker'], True)
            temp_board_sim.scores[AIPlayer.AI_PLAYER_KEY] += move_candidate['score']

            current_eval = AIPlayer.minimax(
                temp_board_sim, AIPlayer.MAX_DEPTH - 1, -math.inf, math.inf, next_is_maximizing)

            if current_eval > best_heuristic_score:
                best_heuristic_score = current_eval
                best_move_info = move_candidate

        elapsed = time.time() - turn_start_time
        logger.info(f"AI evaluation loop finished in {elapsed:.3f}s.")

        if best_move_info:
            logger.info(
                f"AI Chose: '{best_move_info['word']}' (Full: '{best_move_info['full_word_played']}') at ({best_move_info['row']},{best_move_info['col']}) Dir:{best_move_info['direction']}. Score:{best_move_info['score']}. Heuristic:{best_heuristic_score:.2f}.")
            return {"word": best_move_info["word"], "row": best_move_info["row"], "col": best_move_info["col"], "direction": best_move_info["direction"]}
        return None

    @staticmethod
    def minimax(board_state: 'Board', depth: int, alpha: float, beta: float, is_maximizing_player: bool) -> float:
        if depth == 0 or board_state.game_over:
            return AIPlayer.evaluate_board(board_state)

        current_sim_player = AIPlayer.AI_PLAYER_KEY if is_maximizing_player else AIPlayer.HUMAN_PLAYER_KEY
        possible_moves = AIPlayer._get_valid_moves_for_player(
            board_state, current_sim_player)

        if not possible_moves:
            return AIPlayer.evaluate_board(board_state)

        if is_maximizing_player:
            max_eval = -math.inf
            for move in possible_moves:
                child_state = deepcopy(board_state)
                for tile_info in move['placed_info']:
                    child_state.board[tile_info['pos'][0]][tile_info['pos'][1]] = (
                        tile_info['tile_marker'], True)
                child_state.scores[current_sim_player] += move['score']
                child_state.remove_from_rack(
                    [info['tile_marker'] for info in move['placed_info']], current_sim_player)
                child_state.refill_rack(current_sim_player)
                child_state.first_move = False
                child_state.consecutive_passes = 0
                if not child_state.tile_bag and not child_state.player_racks[current_sim_player]:
                    child_state.game_over = True

                double_turn_triggered = any(
                    info['tile_marker'] == POWER_TILE_DOUBLE_TURN_MARKER for info in move['placed_info'])
                next_maximizer = double_turn_triggered
                if not double_turn_triggered:
                    child_state.current_player = AIPlayer.HUMAN_PLAYER_KEY
                else:
                    child_state.current_player = AIPlayer.AI_PLAYER_KEY

                eval_score = AIPlayer.minimax(
                    child_state, depth - 1, alpha, beta, next_maximizer)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:  # Minimizing player (Human)
            min_eval = math.inf
            for move in possible_moves:
                child_state = deepcopy(board_state)
                for tile_info in move['placed_info']:
                    child_state.board[tile_info['pos'][0]][tile_info['pos'][1]] = (
                        tile_info['tile_marker'], True)
                child_state.scores[current_sim_player] += move['score']
                child_state.remove_from_rack(
                    [info['tile_marker'] for info in move['placed_info']], current_sim_player)
                child_state.refill_rack(current_sim_player)
                child_state.first_move = False
                child_state.consecutive_passes = 0
                if not child_state.tile_bag and not child_state.player_racks[current_sim_player]:
                    child_state.game_over = True

                double_turn_triggered = any(
                    info['tile_marker'] == POWER_TILE_DOUBLE_TURN_MARKER for info in move['placed_info'])
                next_maximizer = not double_turn_triggered
                if not double_turn_triggered:
                    child_state.current_player = AIPlayer.AI_PLAYER_KEY
                else:
                    child_state.current_player = AIPlayer.HUMAN_PLAYER_KEY

                eval_score = AIPlayer.minimax(
                    child_state, depth - 1, alpha, beta, next_maximizer)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval

    @staticmethod
    def evaluate_board(board_state: 'Board') -> float:
        ai_score_val = board_state.scores.get(AIPlayer.AI_PLAYER_KEY, 0)
        human_score_val = board_state.scores.get(AIPlayer.HUMAN_PLAYER_KEY, 0)

        if board_state.game_over:
            sim_scores = deepcopy(board_state.scores)  # Simulate on a copy
            sim_racks = deepcopy(board_state.player_racks)

            player_emptied = next((p for p in [
                                  AIPlayer.AI_PLAYER_KEY, AIPlayer.HUMAN_PLAYER_KEY] if not sim_racks[p]), None)
            if player_emptied:
                opponent = AIPlayer.HUMAN_PLAYER_KEY if player_emptied == AIPlayer.AI_PLAYER_KEY else AIPlayer.AI_PLAYER_KEY
                opponent_rack_val = sum(LETTER_SCORES.get(
                    tm, 0) for tm in sim_racks[opponent] if tm != ' ' and tm not in POWER_TILE_TYPES)
                sim_scores[player_emptied] += opponent_rack_val
                sim_scores[opponent] -= opponent_rack_val
            else:  # No one emptied, both lose points for remaining tiles
                for p in [AIPlayer.AI_PLAYER_KEY, AIPlayer.HUMAN_PLAYER_KEY]:
                    rack_val = sum(LETTER_SCORES.get(
                        tm, 0) for tm in sim_racks[p] if tm != ' ' and tm not in POWER_TILE_TYPES)
                    sim_scores[p] -= rack_val
            ai_score_val = sim_scores.get(AIPlayer.AI_PLAYER_KEY, 0)
            human_score_val = sim_scores.get(AIPlayer.HUMAN_PLAYER_KEY, 0)

        score_difference = ai_score_val - human_score_val

        objective_value = 0
        if board_state.player_objectives[AIPlayer.AI_PLAYER_KEY]["completed"]:
            objective_value += 50
        if board_state.player_objectives[AIPlayer.HUMAN_PLAYER_KEY]["completed"]:
            objective_value -= 50

        rack_quality = 0
        ai_rack_sim = board_state.player_racks[AIPlayer.AI_PLAYER_KEY]
        vowels = "AEIOU"
        num_vowels = sum(1 for tm in ai_rack_sim if POWER_TILE_TYPES.get(
            tm, {}).get("display", tm) in vowels)
        num_consonants = len(ai_rack_sim) - num_vowels
        rack_quality -= abs(num_vowels - num_consonants) * 1.5
        if ' ' in ai_rack_sim:
            rack_quality += 10
        if 'S' in ai_rack_sim:
            rack_quality += 5
        if len(ai_rack_sim) == 0 and not board_state.tile_bag:
            rack_quality += 20  # Bonus for going out

        return score_difference + objective_value + rack_quality
