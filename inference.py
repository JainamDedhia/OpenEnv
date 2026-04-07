"""
inference.py — Baseline LLM inference script for Rocket Landing OpenEnv.

Required stdout format:
    [START]
    [STEP] step=<int> decision=<str> reward=<float> reason=<str>
    [END] total_reward=<float>

Environment variables:
    API_BASE_URL   OpenAI-compatible base URL
    MODEL_NAME     Model identifier
    HF_TOKEN       Bearer token / API key
"""

from __future__ import annotations
import os
import json
import sys
from openai import OpenAI
from environment import RocketLandingEnv, Action


# ── Config ────────────────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "openai/gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")

if not HF_TOKEN:
    print(
        "ERROR: HF_TOKEN environment variable is not set.\n"
        "Run: export HF_TOKEN='your-api-key'",
        file=sys.stderr,
    )
    sys.exit(1)

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(obs: object, history: list[dict]) -> str:
    history_str = "\n".join(
        f"  Step {h['step']}: action={h['action']}  reward={h['reward']:.4f}"
        for h in history[-5:]
    ) or "  (no history yet)"

    return f"""You are an autonomous rocket landing controller.

GOAL: Land the rocket safely.
  Safe landing = height <= 2.0 AND velocity between -3.0 and 0.5

PHYSICS (understand this):
  Each step: velocity += thrust - 1.5 (gravity)
             height  += velocity
  increase_thrust adds +3.0 m/s to velocity (braking = good near ground)
  decrease_thrust removes 1.0 m/s (faster fall = bad near ground)
  emergency_burn  adds +5.0 m/s (use ONLY when engine_status = failure)
  stabilize       counteracts wind (use when |wind| > 6)
  maintain        no change

VALID ACTIONS (pick exactly one):
  increase_thrust | decrease_thrust | maintain | stabilize | emergency_burn

DECISION RULES (follow in priority order):
  1. engine_status == "failure"  → emergency_burn
  2. |wind| > 6                  → stabilize
  3. height < 55 AND velocity < -3  → increase_thrust  (start braking early!)
  4. velocity between -3 and 0   → maintain  (perfect descent speed, hold it)
  5. height >= 55                → maintain  (conserve fuel at high altitude)
  6. otherwise                   → increase_thrust

CURRENT STATE:
  height:         {obs.height:.2f} m
  velocity:       {obs.velocity:.2f} m/s  (negative = descending)
  fuel:           {obs.fuel:.2f}
  engine_status:  {obs.engine_status}
  wind:           {obs.wind:.2f} m/s
  step:           {obs.step} / {obs.max_steps}
  last_action:    {obs.last_action}

STEP HISTORY (last 5):
{history_str}

RESPOND WITH JSON ONLY. No markdown, no extra text.
{{"decision": "<action>", "reason": "<one sentence explanation>"}}
"""


# ── LLM caller ────────────────────────────────────────────────────────────────

FALLBACK_SEQUENCE = [
    "increase_thrust", "maintain", "stabilize",
    "maintain", "increase_thrust", "maintain",
]

def get_action(obs: object, history: list[dict], fallback_idx: int) -> tuple[dict, int]:
    prompt = build_prompt(obs, history)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        content = response.choices[0].message.content.strip()

        # Strip markdown fences
        if "```" in content:
            parts = content.split("```")
            content = parts[1] if len(parts) > 1 else parts[0]
            if content.lower().startswith("json"):
                content = content[4:]

        # Extract JSON object
        start = content.find("{")
        end   = content.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON object in response")

        parsed = json.loads(content[start:end])

        if "decision" not in parsed:
            raise ValueError("Missing 'decision' key")

        if parsed["decision"] not in RocketLandingEnv.VALID_ACTIONS:
            raise ValueError(f"Invalid action: {parsed['decision']}")

        return parsed, fallback_idx

    except Exception as exc:
        print(f"[WARN] LLM error: {exc}", file=sys.stderr)
        fallback_action = FALLBACK_SEQUENCE[fallback_idx % len(FALLBACK_SEQUENCE)]
        fallback_idx += 1
        return {
            "decision": fallback_action,
            "reason":   f"fallback-{fallback_action} due to parse error",
        }, fallback_idx


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    env  = RocketLandingEnv()
    obs  = env.reset()
    done = False

    total_reward = 0.0
    history: list[dict] = []
    fallback_idx = 0

    print("[START]")

    while not done:
        output, fallback_idx = get_action(obs, history, fallback_idx)

        action = Action(decision=output["decision"])
        obs, reward, done, _ = env.step(action)

        total_reward += reward.score

        print(
            f"[STEP] "
            f"step={obs.step} "
            f"decision={action.decision} "
            f"reward={reward.score:.4f} "
            f"reason={output.get('reason', 'N/A')}"
        )

        history.append({
            "step":   obs.step,
            "action": action.decision,
            "reward": reward.score,
        })

    print(f"[END] total_reward={round(total_reward, 4)}")


if __name__ == "__main__":
    main()