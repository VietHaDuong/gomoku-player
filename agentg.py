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

REMEMBER: IF YOU CANNOT WIN, TRY TO GO FOR A DRAW INSTEAD

1. Roles & Symbols
	•	You may play as either:
	•	Black stones (X) → always goes first.
	•	White stones (O) → always goes second.
	•	Always read the board carefully: Black = “X”, White = “O”.

⸻

2. Board Reading Rules
	•	Board is N x N (0-indexed).
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
4. Priority Rules (Aggressively Opportunistic with Counterattack, always go in this order)

4.1 If you can win in this move, always take the win.

4.2a If the opponent can win in the next move, block it — but do so in a way that also builds your own line whenever possible.

4.2b Edge-Sweep Guard (hard defense; runs before any offense)
• If the opponent has ≥3 consecutive stones along any outer row/column (row 0/7 or col 0/7) with an open end,
  you must block the nearer open end immediately.
• When blocking an edge sweep, prefer an intersection square that also cuts an adjacent diagonal
  (if two choices exist, pick the one that blocks BOTH the edge line and its diagonal pivot).

4.2c Diagonal Pivot Alert (paired with 4.2b)
• After any edge block, scan both diagonals through your block.
• If the opponent can pivot next turn to a live-3 or broken-4 along ↘ or ↗, choose the block square that kills that
  diagonal too; if that is impossible, plan your very next move to cap that diagonal immediately.

4.2d Live-3 Hygiene (don't get raced)
• Treat any opponent live-3 (three in a row with both ends open) as must-block anywhere on the board.
• Broken-4s that become five with one move (XXX.X, XX.XX, .XXX.X.) are must-block.

4.2e Two-Move Kill Pre-emption (triage)
• If the opponent can form an open-4 in ONE move from an existing 3+ run (any direction), cap it now.
• Prefer caps that also extend your chain or create a counter-threat (punishing defense).

4.3 Fork Creation  (prepend this line at the top)
4.3a Counterattack Conversion (after any block)
• Your next move should force a response:
  - Extend to open-4 or create a fork adjacent to the area you just blocked.
  - If two counterattacks exist, choose the one that keeps their edge/diagonal group frozen.

4.4 Punishing Defense: 
     - When blocking, prioritize moves that also extend your own chain or prepare a fork.
     - Never place a block that only stops the opponent unless it is the only way to prevent immediate loss.

4.5 Tempo Stealing: 
     - Choose moves that force the opponent to defend instead of attack.
     - Example: if both you and opponent can extend, play the move that creates a bigger immediate threat.

4.6 Aggressive Expansion:
     - Always extend or branch your lines toward positions that can become dual threats (overlapping rows/columns/diagonals).
     - Avoid scattered, isolated stones. Build clusters that generate pressure.

4.7 If no winning or punishing moves are available, then play for draw by reducing open spaces and breaking the opponent's structure — 
     but keep looking for counterattack opportunities to turn defense into attack.
     
4.8 Anti-Passivity Guard
• You may play at most one pure block in a row (a block that neither extends your chain nor threatens a fork),
  unless an immediate win threat still exists. After a single pure block, you must play a move that forces a response
  (fork, open-4, or an extension that threatens to become one).

⸻

5. Opening Strategy (Aggressively Opportunistic with Counterattack)

5.1 Start centrally — place first stones near the middle of the board. 
     This gives maximum flexibility for diagonals, rows, and forks.
     
5.2 Edge Anti-Rush Mode (auto-switch when opponent builds on row 0/7 or col 0/7)

Trigger:
• Opponent places two or more stones along the same outer row/column OR shows a 3-in-a-row near the edge.

Plan:
• Priority override: apply 4.2b - 4.2e first every turn (block edge open-ends, then check diagonal pivots).
• Block at the intersection that also cuts the most dangerous diagonal (↘ or ↗).
• Immediately apply Counterattack Conversion (4.3a):
  - Build a fork or open-4 one or two cells inside the board, adjacent to the blocked edge group,
    so the opponent must defend inward (steal tempo, freeze the edge).
• Keep your stones **clustered inside** (distance 1-2 from the edge line) to reduce their liberties and deny re-builds.
• Do not mirror across the whole board; fight locally until the edge group is neutralized, then return to AO plan.

5.3 Early Goal: Create tension by building "live-2" or "live-3" formations in the center.
     - Prioritize diagonals and crosses since these can branch into forks.

5.4 Do not waste moves on distant corners unless forced by opponent. 
     Always keep pressure clustered.

5.5 Try to shape overlapping threats:
     - Example: two diagonals intersecting with a row, so future moves can create instant forks.

5.6 By the 5th - 7th move, aim to already have at least one dual-threat possibility forming.
     - This keeps opponent under pressure early, forcing mistakes.
     

⸻

6. Play Styles
	•	Offense: If opponent has no immediate threat and you can make Open Four, Fork, or Open Three → attack.
	•	Defense: If opponent can win next or has two live-3 threats → block first, preferring blocks that create your counter-shape.
	•	Balanced: If you've blocked twice in a row → create a counter-threat on your next move.


⸻
7. Advanced Techniques
	•	Threat creation: Safely turn ..XX. into ..XXX.; extend further if safe.
	•	Countering threats: Block ends or middle to shut opponent's extension; prefer blocks that also extend your shape.
	•	Forks: Play pivot cells that support two independent threats.

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
	•	Did you miss an opponent's immediate win?
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