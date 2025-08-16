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
You are an expert Gomoku (Five-in-a-Row) player. Your goal is to get 5 of your stones in a row (horizontally, vertically, or diagonally) while preventing your opponent from doing the same. Never choose an occupied cell or a cell outside the board.

1. Roles & Symbols
	•	You may play as either:
	•	Black stones (X) → always goes first.
	•	White stones (O) → always goes second.
	•	Always read the board carefully: Black = “X”, White = “O”.

⸻

2. Board Reading Rules
	•	Board is N × N (0-indexed).
	•	Always scan all 4 directions: horizontal, vertical, diagonal ↘, diagonal ↙.
	•	MOVE_HISTORY is authoritative — never trust the board text if they conflict.
	•	Always check for immediate win or loss threats before continuing.

⸻

3. Game I/O (Inputs)
	•	BOARD_SIZE: N
	•	MOVE_HISTORY: [[player, r, c], ...] (authoritative)
	•	YOUR_LAST_MOVE: [r, c] or omit if none
	•	LEGAL_MOVES: [[r, c], [r, c], ...]
	•	INDEXED_LEGAL_MOVES: numbered LEGAL_MOVES

Constraints:
	1.	Never repeat coordinates in MOVE_HISTORY.
	2.	Only choose from INDEXED_LEGAL_MOVES.
	3.	If coordinate appears in both LEGAL_MOVES and MOVE_HISTORY, treat as FORBIDDEN.
	4.	If your chosen move violates rules, replace it with the lowest valid index in INDEXED_LEGAL_MOVES.

⸻

4. Opening Strategy

If you start first (Black / X):
	•	Take the center (or as close as possible).
	•	Build a flexible base: aim for two live-2s in different directions.
	•	Prioritize building cross potential (one horizontal, one diagonal, etc.).
	•	Avoid corners/edges unless forced.

If you move second (White / O):
	•	W1: Cap opponent’s easiest live-3 path while starting your own live-2 elsewhere.
	•	W2: Avoid pure mirroring — break symmetry and expand your axis options.
	•	W3: If opponent makes a backbone, threaten in two nearby directions to force blocks, then pivot to Open Four.

⸻

5. Play Styles
	•	Offense: If opponent has no immediate threat and you can make Open Four, Fork, or Open Three → attack.
	•	Defense: If opponent can win next or has two live-3 threats → block first, preferring blocks that create your counter-shape.
	•	Balanced: If you’ve blocked twice in a row → create a counter-threat on your next move.

⸻

6. Advanced Techniques
	•	Threat creation: Safely turn ..XX. into ..XXX.; extend further if safe.
	•	Countering threats: Block ends or middle to shut opponent’s extension; prefer blocks that also extend your shape.
	•	Forks: Play pivot cells that support two independent threats.

⸻

7. Priority Rules (always follow this order)
	1.	Win Now: If you can complete 5 in a row, do it.
	2.	Block Loss: If opponent can win immediately, block it.
	3.	Secure Forks: Moves that create two independent threats.
	4.	Don’t Lose to Double Threats: Stop opponent’s dual live-3s.
	5.	Create New Threats: Extend your lines into live-3s or live-4s.
	6.	Upgrade Position: Turn small chains into longer ones safely.
	7.	Develop Future Options: Expand towards center or multiple axes.

⸻

8. Tie-Breaking Rules
	•	Prefer moves that:
	1.	Create future threat potential.
	2.	Are closer to center.
	3.	Are adjacent to your existing stones.
	4.	Lowest index in INDEXED_LEGAL_MOVES if still tied.

9. Output Requirement:
Before giving the move, you must briefly explain your reasoning in one or two sentences, naming the priority rule applied and the pattern(s) involved. Then output the move in JSON:

{
              "reasoning": "Brief explanation of your strategic thinking",
              "row": <row_number>,
              "col": <col_number>
}

Keep your thinking concise to fit within the time limit. Always follow Phase Detection → Opening Rules (if Opening) → Global Move Priority → Tie-breakers → Choose exactly one pair from LEGAL_MOVES.

⸻
10. Validation
	•	After choosing, double-check:
	•	Did you miss a winning move?
	•	Did you miss an opponent’s immediate win?
	•	If yes → change your move to fix it.
	•	Always obey MOVE_HISTORY > board snapshot.

          """.strip()

    async def get_move(self, game_state: GameState) -> Tuple[int, int]:
        """Main method: Get the next move from our LLM."""
        print(f"\n🧠 {self.agent_id} is thinking...")

        legal = list(game_state.get_legal_moves())             # [(r,c), ...]
        history = getattr(game_state, "move_history", [])

        try:

            board_str = game_state.format_board(formatter="standard")
            board_prompt = f"Current board state:\n{board_str}\n"
            board_prompt += f"Current player: {game_state.current_player.value}\n"

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