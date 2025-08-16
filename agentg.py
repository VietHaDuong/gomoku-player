import re
import json
import random
from typing import Tuple
from gomoku import Agent
from gomoku.llm import OpenAIGomokuClient
from gomoku.core.models import Player, GameState, GameResult

class AgentG(Agent):

    def _setup(self):
      self.system_prompt = self._create_system_prompt()
      self.llm_client = OpenAIGomokuClient(api_key='sk-sFuwp_XclDL0Xy3Uakb--w', model='deepseek/deepseek-r1-0528-qwen3-8b', endpoint='https://api.mtkachenko.info/v1')

    def _create_system_prompt(self) -> str:
        """Create the system prompt that teaches the LLM how to play Gomoku."""
        return """
You are a Gomoku agent. You must pick a legal move on every turn.

Ground rules:
- Board uses 0-indexed rows/cols. "X" = Black (moves first), "O" = White.
- Respond with JSON ONLY: { "reasoning": "Brief explanation of your strategic thinking", "row": <int>, "col": <int>}.
- Never choose an occupied cell or a cell outside the board.
- Keep internal reasoning brief; do not include it in the output.

Play policy (You MUST follow this rules EXACTLY):
1) Opening:
   - If there are zero stones, play the exact center: (size//2, size//2). 
   - If there are already stone (from the opponent in the center), play it in the nearest position to opponent stone.
	 - After placing your stone, try to build on it to win (e.g. XXXXX or O if you started as white (second player)).
	 - Concentrate on controlling the center of the board, as this can provide most of the opportunities.
	 - You MUST always try to go for a win move in the center at first (e.g. (3,3), (2,3), (1,3) ,(4,3) ,(5,3) for vertical win. Try to go horizontal, both diagonals too)

2) Midgame priorities (no explicit “win-move list” needed—apply heuristics):
   - Prefer moves that extend your longest line (horizontal/vertical/diagonal) with low blocking.
   - Create threats: aim for open-threes and open-fours when available; prefer placements that produce multiple converging lines.
   - Avoid helping the opponent create immediate strong threats.
   - If the opponent has an immediate line-of-four with an open end, block it.

3) Tie-breakers (in order):
   - Centrality (closer to center).
   - Creates/extends more lines simultaneously.
   - Keeps flexibility (more follow-up empties adjacent).

You have up to 20 seconds to think carefully. Do not answer immediately. First, write your reasoning. Only after your reasoning is complete, choose the final move.

          """.strip()

    def assess_board(self, game_state: GameState) -> dict:
        """
        Return a minimal, LLM-friendly snapshot of the current board state.
        No strategy, no win/opportunity detection — just facts.

        Output schema:
        {
            "bot_side": "X" | "O",
            "opponent_side": "O" | "X",
            "stones": [(row, col, "X"|"O"), ...],
            "empty_positions": [(row, col), ...],
            "size": int   # board is size x size
        }
        """
        # 1) Determine bot side from the game engine if available; fallback to count rule.
        bot_side = None
        try:
            # Many engines expose current_player.value as "X" or "O"
            bot_side = getattr(getattr(game_state, "current_player", None), "value", None)
            if bot_side not in ("X", "O"):
                bot_side = None
        except Exception:
            bot_side = None

        # 2) Get a machine-parseable board by formatting, then parsing.
        # We assume format_board("standard") includes only X/O/. somewhere per line.
        try:
            board_str = game_state.format_board(formatter="standard")
        except Exception:
            # Fallback if other formatter name is required
            board_str = game_state.format_board()

        rows = []
        for line in board_str.splitlines():
            symbols = [ch for ch in line if ch in ("X", "O", ".")]
            if symbols:
                rows.append(symbols)

        if not rows:
            # If formatting gave nothing parseable, provide an empty minimal structure
            return {
                "bot_side": bot_side or "X",  # harmless default
                "opponent_side": "O" if (bot_side or "X") == "X" else "X",
                "stones": [],
                "empty_positions": [],
                "size": 0,
            }

        size = max(len(r) for r in rows)
        # Normalize jagged rows (just in case)
        for i, r in enumerate(rows):
            if len(r) < size:
                rows[i] = r + ["." for _ in range(size - len(r))]
        grid = rows  # size x size of "X"/"O"/"."

        # 3) Collect stones and empties
        stones = []
        empty_positions = []
        countX = countO = 0
        for r in range(len(grid)):
            for c in range(len(grid[r])):
                cell = grid[r][c]
                if cell == ".":
                    empty_positions.append((r, c))
                elif cell in ("X", "O"):
                    stones.append((r, c, cell))
                    if cell == "X":
                        countX += 1
                    else:
                        countO += 1

        # 4) If bot_side still unknown, infer by count rule (X starts first):
        #    - If X has more stones, it's O's turn → bot is "O"
        #    - If O has more stones, it's X's turn → bot is "X"
        if bot_side not in ("X", "O"):
            if countX > countO:
                bot_side = "O"
            elif countO > countX:
                bot_side = "X"
            else:
                # Equal counts → X to move at start of game
                bot_side = "X"

        return {
            "bot_side": bot_side,
            "opponent_side": "O" if bot_side == "X" else "X",
            "stones": stones,
            "empty_positions": empty_positions,
            "size": size,
        }

    async def get_move(self, game_state: GameState) -> Tuple[int, int]:
        """Main method: Get the next move from our LLM."""
        print(f"\n🧠 {self.agent_id} is thinking...")

        legal = list(game_state.get_legal_moves())             # [(r,c), ...]
        history = getattr(game_state, "move_history", [])

        try:

            board_str = game_state.format_board(formatter="standard")

            board_prompt = f"Current board state:\n{board_str}\n"
            board_prompt += f"Current player: {game_state.current_player.value}\n"
            state = self.assess_board(game_state)
            board_prompt += "\nSTATE_JSON (from assess_board):\n" + json.dumps(state, separators=(',', ':')) + "\n"

            # Create messages for the LLM
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"{board_prompt}\n\nPlease provide your next move as JSON."},
            ]

            print("💡 Full Prompt:\n\n")
            print(json.dumps(messages, indent=2, ensure_ascii=False))
            print()

            # Get response from LLM
            response = await self.llm_client.complete(messages)

            print("💡 Response:\n\n")
            print(response)
            print()

            if m := re.search(r"{[^}]+}", response, re.DOTALL):
                json_data = json.loads(m.group(0).strip())
                return json_data["row"], json_data["col"]

        except Exception as e:
            print(e)

        return self._get_fallback_move(game_state)

    def _get_fallback_move(self, game_state: GameState) -> Tuple[int, int]:
        """Simple fallback when LLM fails."""
        return random.choice(game_state.get_legal_moves())

print("🎉 Agent G is defined!")
print("   This agent demonstrates LLM-style strategic thinking.")