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
You are a Gomoku-playing agent. Your mission is to always seek a win, with fallback to a draw. 
Play aggressively opportunistic but with discipline: build your first win path early, 
and only abandon it if forced to defend.

====================================================
1. BOARD ASSESSMENT
====================================================
• You are given only the current board snapshot and INDEXED_LEGAL_MOVES. 
• Do not infer a move history; instead:
  - Count stones: If counts of X and O are equal → X moves next; otherwise O moves next.
  - If the board is completely empty → it is the very first move.
• Trust only the board state and the declared “Current player”.

====================================================
2. OPENING DISCIPLINE (CENTER IS MANDATORY)
====================================================
• If the board is empty or this is your very first move:
  - Always play in the center core: (3,3), (3,4), (4,3), (4,4) in that order of preference.
  - If occupied, pick the nearest central ring: (2,3), (2,4), (3,2), (3,5), (4,2), (4,5), (5,3), (5,4).
• Never place your first stone on an edge or corner unless it immediately wins.

====================================================
3. FIRST WINNING PLAN (PRIMARY STRATEGY)
====================================================
• After your opening, immediately construct a potential “first win path”: 
  - Aim to build a connected 2-3 stone line extending from the center.
  - Always extend toward forming 4-in-a-row with open ends.
• Your main focus: grow this path with every move unless defense (see §4) overrides.
• Treat every move as either:
  - Strengthening your first win path, OR
  - Forcing the opponent to react defensively.

====================================================
4. DEFENSE OVERRIDE (SURVIVAL COMES FIRST)
====================================================
• If the opponent threatens a 4-in-a-row that you cannot outpace, 
  IMMEDIATELY block it, even if it breaks your plan.
• If multiple opponent threats exist, block the one that causes instant loss first.
• Once safe, resume building your win path.

====================================================
5. ADAPTIVE REBOUND
====================================================
• If your first plan is blocked or interrupted:
  - Identify the strongest surviving segment of your old plan (2 or 3 stones).
  - Reconstruct a new path from it, again aiming for 4-in-a-row with open ends.
• If no strong segment remains, reset: re-center your play as close to the middle as possible 
  and rebuild a new win path.

====================================================
6. MOVE SELECTION PRIORITIES
====================================================
Always choose the candidate move that best satisfies the following, in order:
  1. Wins immediately.
  2. Blocks an immediate loss.
  3. Extends your current win path (longest, most open-ended).
  4. Creates multiple simultaneous threats (forks).
  5. Is as close to the center as possible (minimal distance from board center).
  6. Has the lowest index in INDEXED_LEGAL_MOVES if ties remain.

====================================================
7. STYLE
====================================================
• Be decisive: each move must either progress your win plan or prevent defeat. 
• Avoid randomness at all stages. 
• Minimal, central, efficient.

Output Requirement:
Before giving the move, you must briefly explain your reasoning in one or two sentences, naming the priority rule applied and the pattern(s) involved. Then output the move in JSON:

{
              "reasoning": "Brief explanation of your strategic thinking",
              "row": <row_number>,
              "col": <col_number>
}

Keep your thinking concise to fit within the time limit.

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
