"""
inference.py — Baseline inference script for Rocket Landing OpenEnv.

Stdout format (strict — do not modify field names or order):
    [START]
    [STEP] step=<int> decision=<str> reward=<float> reason=<str>
    [END] total_reward=<float>

Environment variables required:
    API_BASE_URL   Base URL of the OpenAI-compatible LLM endpoint
    MODEL_NAME     Model identifier (e.g. "openai/gpt-4o-mini")
    HF_TOKEN       Hugging Face / API key passed as the bearer token
"""

import os
import json
import sys
from openai import OpenAI
from environment import RocketLandingEnv, Action

# ────────────────────────────────────────────────────────────────────────────
# 1. Configuration — read from environment variables (required by spec)
# ────────────────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "openai/gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "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")

if not HF_TOKEN:
    print(
        "WARNING: HF_TOKEN is not set. "
        "Set the HF_TOKEN environment variable before running.",
        file=sys.stderr,
    )

# ────────────────────────────────────────────────────────────────────────────
# 2. OpenAI-compatible client (required by spec)
# ────────────────────────────────────────────────────────────────────────────
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

# ────────────────────────────────────────────────────────────────────────────
# 3. Helper — build the strategic prompt
# ────────────────────────────────────────────────────────────────────────────
def build_prompt(obs, history: list) -> str:
    history_str = "\n".join(
        f"Step {h['step']}: {h['action']} -> reward={h['reward']:.2f}"
        for h in history[-3:]
    ) or "None"

    repeat_warning = ""
    if len(history) >= 2 and history[-1]["action"] == history[-2]["action"]:
        repeat_warning = "WARNING: You repeated the same action twice. You MUST change strategy now."

    return f"""You are an advanced autonomous rocket landing AI controller.

MISSION GOAL:
- Guide the rocket to land safely: height < 10, velocity between -3 and 0.
- Conserve fuel and avoid engine failures.

VALID ACTIONS (choose exactly one):
  increase_thrust | decrease_thrust | maintain | stabilize | emergency_burn

DECISION PRIORITY (follow in order):
1. engine_status == "failure"  → emergency_burn
2. abs(wind) > 6               → stabilize
3. height > 160                → decrease_thrust
4. height < 60                 → increase_thrust
5. abs(velocity) < 5           → maintain
6. otherwise                   → decrease_thrust

CRITICAL RULES:
- NEVER repeat the same action consecutively — you will be penalised.
- After emergency_burn, switch to stabilize or maintain.
- Think step by step before deciding.

CURRENT STATE:
  Height:         {obs.height:.2f} m
  Velocity:       {obs.velocity:.2f} m/s
  Fuel:           {obs.fuel:.2f}
  Engine status:  {obs.engine_status}
  Wind:           {obs.wind:.2f} m/s
  Step:           {obs.step} / 6
  Last action:    {obs.last_action}

RECENT HISTORY (last 3 steps):
{history_str}

{repeat_warning}

RESPOND WITH VALID JSON ONLY — no markdown, no extra text.
FORMAT:
{{"decision":"<action>","reason":"<one short sentence>","predicted_next":{{"height_trend":"up|down","velocity_trend":"faster|slower"}}}}
"""


# ────────────────────────────────────────────────────────────────────────────
# 4. Helper — call the LLM and parse the response robustly
# ────────────────────────────────────────────────────────────────────────────
FALLBACK_ACTION = "maintain"

def get_action(obs, history: list) -> dict:
    prompt = build_prompt(obs, history)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
        )
        content = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        if "```" in content:
            parts = content.split("```")
            content = parts[1] if len(parts) > 1 else parts[0]
            if content.startswith("json"):
                content = content[4:]

        # Extract the first valid JSON object
        start = content.find("{")
        end   = content.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON object found in response")

        parsed = json.loads(content[start:end])

        if "decision" not in parsed:
            raise ValueError("Missing 'decision' key in JSON response")

        # Guard against hallucinated actions
        if parsed["decision"] not in RocketLandingEnv.VALID_ACTIONS:
            raise ValueError(f"Unknown action: {parsed['decision']}")

        return parsed

    except Exception as exc:
        print(f"[WARN] LLM parse error: {exc} — using fallback action '{FALLBACK_ACTION}'",
              file=sys.stderr)
        return {
            "decision": FALLBACK_ACTION,
            "reason": "fallback due to parse error",
            "predicted_next": {"height_trend": "down", "velocity_trend": "slower"},
        }


# ────────────────────────────────────────────────────────────────────────────
# 5. Main loop
# ────────────────────────────────────────────────────────────────────────────
def main():
    env     = RocketLandingEnv()
    obs     = env.reset()
    done    = False
    total_reward = 0.0
    history: list[dict] = []

    # Required by spec — must be the very first line of stdout
    print("[START]")

    while not done:
        output = get_action(obs, history)

        action              = Action(decision=output["decision"])
        obs, reward, done, _ = env.step(action)

        total_reward += reward.score

        # Required [STEP] log format — field names must match exactly
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

    # Required [END] log format — must be the very last line of stdout
    print(f"[END] total_reward={round(total_reward, 4)}")


if __name__ == "__main__":
    main()