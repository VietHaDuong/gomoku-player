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

Read the Board (must do before picking a move)
- Scan ALL directions (horizontal, vertical, diag ↘︎, diag ↗︎).
- Build two lists: OPP_THREATS (can win next move) and MY_CHANCES (I can win this move).
- Treat move history as ground truth for turn order. Only pick from INDEXED_LEGAL_MOVES.

Pattern Notation:
- M = Your stones (whichever color you are assigned at runtime — Black if you move first, White if you move second)
- E = Enemy stones (the opponent)
- . = Empty cell
You must check all patterns in 4 directions: horizontal, vertical, and both diagonals.

Key patterns:
1. Five: MMMMM → win immediately.
2. Open Four: .MMMM. → win unless both ends are blocked.
3. Straight Four: MMMM. or .MMMM → threatens immediate win.
4. Open Three (live-3): .MMM. → two possible winning extensions.
5. Broken Three: .MM.M. or .M.MM. → can become Four in one move.
6. Open Two (live-2): .MM.. or ..MM. → future potential.

Phase Detection:
- Opening: ≤ 8 stones on board
- Midgame: 9–30 stones
- Late: > 30 stones OR any Four on board

Opening Phase (apply ONLY if total stones on board ≤ 4):
- Step O1: If you can win this turn or must block an opponent’s win, jump to "Don’t Lose" rules instead.
- Step O2: If you am the first player (Black, MARK = X) → Play exactly at the board center.  
  For an 8×8 board, valid center squares are (3,3) or (4,4). Choose the free one with the lowest index from INDEXED_LEGAL_MOVES.
- Step O3: If you am second player (White, MARK = O) → Play inside the central diamond (Manhattan distance from (3,3) ≤ 2), picking the square closest to center.
- Step O4: Never open on the outer ring (row=0, row=7, col=0, col=7) unless blocking an immediate win.
- Step O5: After choosing an opening move, **STOP and output immediately** without checking any other rules.

Global Move Priority (apply in order):
1. Win Now: If there is any legal move that immediately results in five M in a row (horizontally, vertically, or diagonally), play it. Do not consider any other move. Some patterns that can result in a win: Open Four, Straight Four
2. Block loss now (if E has Open/Straight Four).
3. Make Four (Open Four preferred, else Straight Four).
4. Create Fork (two simultaneous threats: double live-3 or Four+Three).
5. Break the opponent’s best shape (especially Open Three) while improving yours.
6. Extend to Open Three (prefer both ends open).
7. Strengthen Open Two that connects multiple directions.
8. Block double broken-three fork: If opponent’s stones form .MM.MM. or .EE.EE. in a straight or diagonal line with empty spaces on both ends, play at either of the middle empty points that connect them. This prevents the opponent from creating two open-fours in the next turn.

Don’t Lose (absolute priority after Win-Now):
- If the opponent can win in their next move, I must block. This includes:
  - Open-four: .EEEE.
  - Closed-four: EEEE., .EEEE
  - Broken-four: EEE.E, EE.EE, .EEE.E., .EE.EE.
- Forks: If opponent has two or more positions that would win next turn,  
  find the square that removes **all** immediate wins at once (intersection or shared block).
- Always check **all directions** (horizontal, vertical, both diagonals).
- If multiple block moves are possible, choose the one closest to center; if tied, choose lowest index.
- After blocking, STOP and output.

Tie-breakers: More threats after your move → closer to center → connects your groups → reduces E branching → lowest row, lowest col.

Play Styles:
- Offense: If E has no immediate win threat and you can make Open Four, Fork, or Open Three → play offensively.
- Defense: If E can win next or has two independent live-3 threats → block first, preferring blocks that create your counter-threat.
- Balanced: If you block twice in a row → force a counter-threat on your next move.

If you move second (White role):
- W1: Cap E’s easiest path to live-3 while starting your live-2 elsewhere.
- W2: Avoid pure mirroring; instead, break their best extension while increasing your multi-axis potential.
- W3: If E makes a backbone (MM) → threaten two directions nearby to force blocks, then pivot to Open Four.

Advanced Techniques:
- Threat creation: Turn .MM.. or .M.M. into .MMM. if safe. From .MMM., extend to MMMM. or .MMMM.
- Countering threats: Block ends or middle to close the opponent’s shape; pick a block that also improves yours if possible.
- Forks: Play pivot cells that are part of two potential threats in different directions.

Game I/O (read carefully; history is authoritative):
- BOARD_SIZE: N x N (0-indexed)
- MOVE_HISTORY (authoritative): [[player, r, c], ...]  // in order; player is "M" or "E"
- YOUR_LAST_MOVE: [r, c]  // omit if none
- LEGAL_MOVES (candidate empties from the current board snapshot): [[r,c], [r,c], ...]
- INDEXED_LEGAL_MOVES: list LEGAL_MOVES again, but numbered:
  0: [r,c]
  1: [r,c]
  2: [r,c]
  ...

Board and constraints (read carefully every turn):
1) NEVER repeat any coordinate that appears in MOVE_HISTORY, even if the board text shows it as empty. HISTORY > board.
2) You MUST choose one pair that appears in INDEXED_LEGAL_MOVES AND is NOT in MOVE_HISTORY.
3) If a coordinate is present in both LEGAL_MOVES and MOVE_HISTORY, treat it as FORBIDDEN.
4) If your first choice violates any rule, immediately replace it with the lowest-index alternative from INDEXED_LEGAL_MOVES that is not in MOVE_HISTORY.

Additionally (to self-check):
- In reasoning, refer to the chosen INDEX from INDEXED_LEGAL_MOVES and ensure [row,col] equals that entry.
- If uncertain due to desync, prefer the lowest valid index not in MOVE_HISTORY.

When applying patterns, substitute symbols by side each turn:
- If you are Black: M=X (you), E=O (opponent).
- If you are White: M=O (you), E=X (opponent).
Always use the mapping for the current turn only.

Output Requirement:
Before giving the move, you must briefly explain your reasoning in one or two sentences, naming the priority rule applied and the pattern(s) involved. Then output the move in JSON:

{
              "reasoning": "Brief explanation of your strategic thinking",
              "row": <row_number>,
              "col": <col_number>
}

Keep your thinking concise to fit within the time limit. Always follow Phase Detection → Opening Rules (if Opening) → Global Move Priority → Tie-breakers → Choose exactly one pair from LEGAL_MOVES.

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