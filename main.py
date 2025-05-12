import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from game_logic.board import Board
from game_logic.ai_player import AIPlayer
from game_logic.utils import VALID_WORDS, initialize_dictionary
from models import GameStateResponse, MoveRequest

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Advanced Scrabble AI Game Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173",
                   "http://127.0.0.1:5173", "https://scrabble-sandy.vercel.app"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"], expose_headers=["*"]
)

initialize_dictionary()
game_board = Board()


@app.get("/api/game/start", response_model=GameStateResponse, tags=["Game Flow"])
async def start_game():
    """Starts a new game, resetting the board state."""
    global game_board
    game_board = Board()  # Re-initialize for a new game
    state = game_board.get_state()
    return GameStateResponse(
        board=state["board"], scores=state["scores"], current_player=state["current_player"],
        player_rack=state["player_rack"], game_over=state["game_over"],
        message="New game started. Your turn!", first_move=state["first_move"],
        tiles_in_bag=state["tiles_in_bag"], human_objective=state["human_objective"]
    )


@app.post("/api/game/move", response_model=GameStateResponse, tags=["Game Actions"])
async def player_move(move: MoveRequest):
    """Handles a move attempt by the human player, followed by the AI's turn if applicable."""
    global game_board
    player = "human"

    if game_board.game_over:
        raise HTTPException(status_code=400, detail="Game is over.")
    if game_board.current_player != player:
        raise HTTPException(status_code=400, detail="Not your turn.")

    human_success, human_msg, _ = game_board.place_word(
        move.word, move.row, move.col, move.direction, player)

    if not human_success:
        state = game_board.get_state()
        return GameStateResponse(
            board=state["board"], scores=state["scores"], current_player=state["current_player"],
            player_rack=state["player_rack"], game_over=state[
                "game_over"], message=f"Invalid Move: {human_msg}",
            first_move=state["first_move"], tiles_in_bag=state["tiles_in_bag"], human_objective=state["human_objective"]
        )

    ai_msg = ""
    if not game_board.game_over and game_board.current_player == "ai":
        logger.info("AI turn starting after human move.")
        # Pass the current game_board instance
        ai_move_details = AIPlayer.get_best_move(game_board)
        if ai_move_details:
            ai_success, msg, _ = game_board.place_word(
                ai_move_details["word"], ai_move_details["row"], ai_move_details["col"], ai_move_details["direction"], "ai")
            ai_msg = msg if ai_success else f"AI Error ({msg}). AI Passes."
            if not ai_success:
                game_board.pass_turn("ai")
        else:
            _, msg = game_board.pass_turn("ai")
            ai_msg = f"AI Passes. ({msg})"
            logger.info("AI chose to pass or found no moves.")

    combined_message = f"You: {human_msg}"
    if ai_msg:
        combined_message += f" || AI: {ai_msg}"
    if game_board.game_over and "GAME OVER" not in combined_message.upper():
        combined_message += f" || GAME OVER! Final Score -> You: {game_board.scores.get('human',0)}, AI: {game_board.scores.get('ai',0)}"

    state = game_board.get_state()
    return GameStateResponse(
        board=state["board"], scores=state["scores"], current_player=state["current_player"],
        player_rack=state["player_rack"], game_over=state["game_over"], message=combined_message,
        first_move=state["first_move"], tiles_in_bag=state["tiles_in_bag"], human_objective=state["human_objective"]
    )


@app.post("/api/game/pass", response_model=GameStateResponse, tags=["Game Actions"])
async def player_pass():
    """Handles the human player passing their turn, followed by the AI's turn if applicable."""
    global game_board
    player = "human"

    if game_board.game_over:
        raise HTTPException(status_code=400, detail="Game is over.")
    if game_board.current_player != player:
        raise HTTPException(status_code=400, detail="Not your turn.")

    _, human_pass_msg = game_board.pass_turn(player)

    ai_msg = ""
    if not game_board.game_over and game_board.current_player == "ai":
        logger.info("AI turn starting after human pass.")
        ai_move_details = AIPlayer.get_best_move(game_board)
        if ai_move_details:
            ai_success, msg, _ = game_board.place_word(
                ai_move_details["word"], ai_move_details["row"], ai_move_details["col"], ai_move_details["direction"], "ai")
            ai_msg = msg if ai_success else f"AI Error ({msg}). AI Passes."
            if not ai_success:
                game_board.pass_turn("ai")
        else:
            _, msg = game_board.pass_turn("ai")
            ai_msg = f"AI Passes. ({msg})"
            logger.info("AI chose to pass or found no moves.")

    combined_message = f"You: {human_pass_msg}"
    if ai_msg:
        combined_message += f" || AI: {ai_msg}"
    if game_board.game_over and "GAME OVER" not in combined_message.upper():
        combined_message += f" || GAME OVER! Final Score -> You: {game_board.scores.get('human',0)}, AI: {game_board.scores.get('ai',0)}"

    state = game_board.get_state()
    return GameStateResponse(
        board=state["board"], scores=state["scores"], current_player=state["current_player"],
        player_rack=state["player_rack"], game_over=state["game_over"], message=combined_message,
        first_move=state["first_move"], tiles_in_bag=state["tiles_in_bag"], human_objective=state["human_objective"]
    )


@app.get("/api/game/state", response_model=GameStateResponse, tags=["Game Info"])
async def get_current_game_state():
    """Retrieves the current game state without making any changes."""
    global game_board
    state = game_board.get_state()
    return GameStateResponse(
        board=state["board"], scores=state["scores"], current_player=state["current_player"],
        player_rack=state["player_rack"], game_over=state["game_over"],
        message="Current game state retrieved.", first_move=state["first_move"],
        tiles_in_bag=state["tiles_in_bag"], human_objective=state["human_objective"]
    )

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Advanced Scrabble backend server...")
    if not VALID_WORDS or len(VALID_WORDS) < 50:
        logger.critical(
            "Word dictionary issue: Dictionary is too small or not loaded. Exiting.")
        exit(1)
    uvicorn.run("main:app", host="0.0.0.0", port=8000,
                reload=True)
