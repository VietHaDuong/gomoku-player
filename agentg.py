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
4. Priority Rules — “Win-first plan, drop it only to survive”

4.1 Win-Now (hard stop)
	•	Scan every legal move. If any move makes five-in-a-row (horizontal, vertical, ↘, ↗), play it and STOP.

4.2 Block-Loss (hard stop)
	•	If opponent can win next move (open/semi-open four; broken four like XXXX., .XXXX, XXX.X, XX.XX) → block the square that removes all immediate wins.
	•	Treat live-3 with both ends open as urgent if your pass gives them an open four.

4.3 Plan-First Progress (when 4.1/4.2 didn't trigger)
	•	Advance your Main Winning Line (defined in §5.3) toward the next milestone (see §5.4).
	•	Simple progress score → choose the highest:
+5 create open-4 .XXXX.
+4 create a fork (two independent winning threats next turn)
+3 upgrade to live-3 .XXX.
+2 extend live-2 to closed-3 XX. (or symmetrical)
+1 extend adjacent toward the center anchor

4.4 Plan-Preserving Defense (when a block is required)
	•	If you must block, pick the block that also extends or protects your Main/Secondary Line.
	•	If no such block exists, place the minimum pure block, then immediately return to §4.3 next move.

4.5 Drop-Plan Triggers (when to abandon/rotate the plan)
	•	Only switch the plan if your Main Line is hard-capped at both ends, or the opponent's block creates a strictly better fork elsewhere.
	•	Otherwise, stay committed; finishing one planned line beats starting over.

4.6 Minimal & Central Bias
	•	Prefer moves that are: (a) closer to the center anchor, (b) adjacent to your stones, (c) cut more opponent threats with the same move.
	•	Avoid outer ring (row/col 0 or 7) unless it's a win or forced block.

4.7 Tie-breakers
	1.	Removes more opponent threats with one move
	2.	Closer to center (see §5.1)
	3.	More adjacency to your cluster
	4.	Lowest index in INDEXED_LEGAL_MOVES

4.8 Validation (before output)
	•	If a Win-Now move exists and your choice isn't it → change to Win-Now.
	•	If opponent still has a one-move win after your choice → change to the block that removes all wins.
	•	Confirm you scanned all 4 directions (horizontal, vertical, ↘, ↗).
     
⸻

5. Opening — “Start in the middle, decide the win route early”

5.1 Middle of an 8x8 board
	•	Core center cells: (3,3) (3,4) (4,3) (4,4)
	•	Center ring (preferred next): (2,3) (2,4) (3,2) (3,5) (4,2) (4,5) (5,3) (5,4)
	•	When choosing “nearest to center,” use Manhattan distance to this set; tie → lowest index in INDEXED_LEGAL_MOVES.

5.2 First two own moves (center discipline)
	•	Your first legal move should be a core center if free; otherwise choose from the center ring closest to core.
	•	Your second own move must still be inside core + center ring unless §4.2 forces a block.
	•	Avoid edges/corners early unless blocking or winning.

5.3 Declare your Main Winning Line (direction + anchor)
	•	On your first/second own move, choose a direction from {horizontal, vertical, ↘, ↗} that passes through/near the center.
	•	Anchor = one of your stones in core/center ring; build outward from here.
	•	Also seed a Secondary Line ≈90° to the Main Line for future forks.

5.4 Milestones to finish the game
	•	M1: reach a connected live-2 along the Main Line
	•	M2: upgrade to live-3 .XXX. along that line
	•	M3: branch to create a fork with the Secondary Line
	•	M4: convert to open-4 .XXXX. → win next turn

5.5 Plan-preserving rule
	•	Every opening move should either:
(a) advance the Main Line toward the next milestone, or
(b) seed/strengthen the Secondary Line near the anchor.
	•	If forced to block, apply §4.4 and then resume Main Line progress.

5.6 Scanning discipline (never skip)
	•	Every evaluation scans all 4 directions; give diagonals equal priority to straight lines.
	•	MOVE_HISTORY is authoritative. Only output a coordinate from INDEXED_LEGAL_MOVES that is not in MOVE_HISTORY.

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
