from fastapi import FastAPI
from models import MoveRequest, MoveResponse, AIvsAIResponse
from game_logic.board import Board
from game_logic.ai_player import get_ai_move

app = FastAPI()
board = Board()

@app.get("/start")
def start_game():
    board.reset()
    return {"message": "Game started", "board": board.get_state()}

@app.post("/move")
def player_move(move: MoveRequest):
    if not board.place_word(move.word, move.row, move.col, move.direction, is_ai=False):
        return {"error": "Invalid move"}
    
    ai_move = get_ai_move(board)
    if ai_move:
        board.place_word(ai_move["word"], ai_move["row"], ai_move["col"], ai_move["direction"], is_ai=True)
    
    return MoveResponse(
        player_board=board.get_state(),
        ai_word=ai_move["word"],
        ai_row=ai_move["row"],
        ai_col=ai_move["col"],
        ai_direction=ai_move["direction"]
    )

@app.get("/ai-vs-ai", response_model=AIvsAIResponse)
def ai_vs_ai_game(max_turns: int = 10):
    board.reset()
    history = []
    turn = 0
    while turn < max_turns:
        move = get_ai_move(board)
        if not move:
            break
        board.place_word(move["word"], move["row"], move["col"], move["direction"], is_ai=True)
        history.append(move)
        turn += 1
    return AIvsAIResponse(
        final_board=board.get_state(),
        moves=history
    )
