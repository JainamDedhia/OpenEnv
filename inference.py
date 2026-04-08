"""
inference.py — Baseline LLM inference script for Rocket Landing OpenEnv.
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

TASK_NAME  = "rocket-landing"
BENCHMARK  = "rocket-landing-openenv"
MAX_STEPS  = 15
SUCCESS_SCORE_THRESHOLD = 0.6
MAX_TOTAL_REWARD = MAX_STEPS

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required.")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


# ── Logging helpers (matches official spec exactly) ───────────────────────────
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error):
    done_str  = str(done).lower()
    error_str = str(error) if error is not None else "null"
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: list):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


# ── Prompt builder ────────────────────────────────────────────────────────────
def build_prompt(obs, history: list[str]) -> str:
    history_str = "\n".join(history[-5:]) or "  (no history yet)"
    return f"""You are an autonomous rocket landing controller.

GOAL: Land the rocket safely.
  Safe landing = height <= 2.0 AND velocity between -3.0 and 0.5

VALID ACTIONS (pick exactly one):
  increase_thrust | decrease_thrust | maintain | stabilize | emergency_burn

DECISION RULES:
  1. engine_status == "failure"       → emergency_burn
  2. |wind| > 6                       → stabilize
  3. height < 55 AND velocity < -3    → increase_thrust
  4. velocity between -3 and 0        → maintain
  5. height >= 55                     → maintain
  6. otherwise                        → increase_thrust

CURRENT STATE:
  height:         {obs.height:.2f} m
  velocity:       {obs.velocity:.2f} m/s
  fuel:           {obs.fuel:.2f}
  engine_status:  {obs.engine_status}
  wind:           {obs.wind:.2f} m/s
  step:           {obs.step} / {obs.max_steps}

HISTORY:
{history_str}

RESPOND WITH JSON ONLY:
{{"decision": "<action>", "reason": "<one sentence>"}}
"""


# ── LLM caller ────────────────────────────────────────────────────────────────
FALLBACK_SEQUENCE = [
    "increase_thrust", "maintain", "stabilize",
    "maintain", "increase_thrust", "maintain",
]

def get_action(obs, history: list[str], fallback_idx: int) -> tuple[dict, int]:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": build_prompt(obs, history)}],
            temperature=0.0,
            max_tokens=200,
        )
        content = response.choices[0].message.content.strip()
        if "```" in content:
            parts = content.split("```")
            content = parts[1] if len(parts) > 1 else parts[0]
            if content.lower().startswith("json"):
                content = content[4:]
        start = content.find("{")
        end   = content.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON in response")
        parsed = json.loads(content[start:end])
        if "decision" not in parsed:
            raise ValueError("Missing decision key")
        if parsed["decision"] not in RocketLandingEnv.VALID_ACTIONS:
            raise ValueError(f"Invalid action: {parsed['decision']}")
        return parsed, fallback_idx
    except Exception as exc:
        print(f"[DEBUG] LLM error: {exc}", file=sys.stderr, flush=True)
        action = FALLBACK_SEQUENCE[fallback_idx % len(FALLBACK_SEQUENCE)]
        fallback_idx += 1
        return {"decision": action, "reason": f"fallback due to: {exc}"}, fallback_idx


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    env     = RocketLandingEnv()
    obs     = env.reset()
    done    = False
    rewards = []
    history = []
    fallback_idx  = 0
    steps_taken   = 0
    success       = False
    score         = 0.0

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            output, fallback_idx = get_action(obs, history, fallback_idx)
            action = Action(decision=output["decision"])

            obs, reward, done, _ = env.step(action)

            error = None
            rewards.append(reward.score)
            steps_taken = step

            log_step(
                step=step,
                action=action.decision,
                reward=reward.score,
                done=done,
                error=error,
            )

            history.append(
                f"Step {step}: {action.decision!r} -> reward {reward.score:+.2f}"
            )

        score   = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score   = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", file=sys.stderr, flush=True)
        score   = sum(rewards) / MAX_TOTAL_REWARD if rewards else 0.0
        score   = min(max(score, 0.0), 1.0)
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()