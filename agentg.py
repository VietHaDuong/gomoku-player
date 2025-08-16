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

4.2 If the opponent can win in the next move, block it — but do so in a way that also builds your own line whenever possible.
4.2a Plan-First Progress (when no win/loss is immediate)
• Play the move that increases progress toward your Main Line milestones the most.
  Simple threat score (pick the highest):
  +5 create open-4 (.XXXX.)
  +4 create fork (two independent threats next turn)
  +3 upgrade to live-3 (.XXX.)
  +2 extend live-2 to closed-3 (XX.)
  +1 extend toward center/Anchor adjacency
  Tie-breakers: more threats cut from opponent > closer to center > lower index.

4.2b Plan-Preserving Defense (when a block is required)
• Choose the block that ALSO (i) extends your Main/Secondary Line or (ii) keeps both lines intact.
• If no such block exists, place the minimal pure block that prevents the loss, then immediately resume Plan-First Progress.

4.2c Re-evaluate Triggers (when to change the plan)
• Switch or rotate the Main Line only if:
  - the line is hard-capped at both ends, or
  - opponent's block creates a strictly better fork elsewhere.
• Otherwise, stay committed—finishing one planned line beats starting over.

4.3 Always prefer moves that create forks (two or more simultaneous winning threats). 
     - For example: extend a chain that also leaves an open diagonal. 
     - If opponent blocks one, the other remains.

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

4.8 Counterattack Bias:
• After any forced block, your next move should create a threat that compels a response (fork or open-4) near your Anchor cluster.
     
⸻

5. Opening Strategy (Aggressively Opportunistic with Counterattack)

5.1 Start centrally — place first stones near the middle of the board, the center of the board is defined as around (3, 4), (3, 3), (4, 3) and (4, 4). 
     This gives maximum flexibility for diagonals, rows, and forks.
     
5.2 Winning-Line Commitment (decide your win route early)

• On your first 1-2 own moves, DECLARE a Main Line:
  - Pick a direction from {horizontal, vertical, ↘, ↗} that goes through/near center.
  - Choose an Anchor cell on that line (one of your existing stones near center).

• Milestones for the Main Line:
  M1: reach a connected live-2 along the line.
  M2: upgrade to a live-3 (.XXX.) along the line.
  M3: branch to a Secondary Line ≈90° to set up a fork.
  M4: convert to open-4 (.XXXX.) → win next.

• Move selection (opening, unless survival triggers):
  - Prefer moves that advance the Main Line toward the next milestone.
  - If two moves advance equally, pick the one that also seeds the Secondary Line.
  - Avoid scattering: every move should extend/branch from your Anchor cluster.

• Plan-preserving defense (opening):
  - If you must block, choose a block that also advances or protects your Main/Secondary Line when possible.
  - If a pure block is unavoidable, return to advancing the Main Line on the very next move.

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
