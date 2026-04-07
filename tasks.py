"""
tasks.py — Graded task definitions for the Rocket Landing OpenEnv environment.

Each grader receives (env, action) and returns a float score in [0.0, 1.0].

Task difficulty progression
────────────────────────────
easy   → 0.0 – 1.0  (valid action selection)
medium → 0.0 – 1.0  (height-aware thrust decisions)
hard   → 0.0 – 1.0  (multi-factor emergency + wind + velocity)
"""

from environment import RocketLandingEnv, Action

VALID_ACTIONS = RocketLandingEnv.VALID_ACTIONS


# ────────────────────────────────────────────────────────────────────────────
# Task 1 — Easy
# Grader: Did the agent choose a valid action from the allowed action space?
# ────────────────────────────────────────────────────────────────────────────
def task_easy(env: RocketLandingEnv, action: Action) -> float:
    """
    Score 1.0 if the agent selected any valid action, 0.0 otherwise.
    Tests basic interface compliance and action-space awareness.
    """
    return 1.0 if action.decision in VALID_ACTIONS else 0.0


# ────────────────────────────────────────────────────────────────────────────
# Task 2 — Medium
# Grader: Did the agent choose the height-appropriate thrust action?
# ────────────────────────────────────────────────────────────────────────────
def task_medium(env: RocketLandingEnv, action: Action) -> float:
    """
    Score 1.0  → optimal height-aware decision
    Score 0.5  → neutral (maintain) when no strong height signal
    Score 0.0  → wrong thrust direction for current altitude

    Tests whether the agent understands altitude-based control strategy.
    """
    s = env.state()
    h = s["height"]

    # High altitude: rocket needs to slow descent → decrease thrust
    if h > 150:
        if action.decision == "decrease_thrust":
            return 1.0
        if action.decision == "maintain":
            return 0.5
        return 0.0

    # Low altitude: rocket needs to brake hard → increase thrust
    if h < 50:
        if action.decision == "increase_thrust":
            return 1.0
        if action.decision == "maintain":
            return 0.5
        return 0.0

    # Mid-range altitude — any reasonable action is acceptable
    if action.decision in ("maintain", "stabilize", "decrease_thrust", "increase_thrust"):
        return 0.7
    return 0.3


# ────────────────────────────────────────────────────────────────────────────
# Task 3 — Hard
# Grader: Multi-factor situational awareness
# ────────────────────────────────────────────────────────────────────────────
def task_hard(env: RocketLandingEnv, action: Action) -> float:
    """
    Score is a weighted sum (capped at 1.0) of three independent signals:

      +0.5  engine failure correctly handled  (emergency_burn)
      +0.3  high wind correctly handled       (stabilize)
      +0.2  low velocity correctly maintained (maintain)

    Tests whether the agent can juggle multiple emergency conditions
    simultaneously and prioritise the most critical one.
    """
    s = env.state()
    score = 0.0

    # Priority 1 — Engine failure must be addressed immediately
    if s["engine_status"] == "failure":
        if action.decision == "emergency_burn":
            score += 0.5
        else:
            score -= 0.1   # penalty for ignoring failure

    # Priority 2 — High cross-wind destabilises the rocket
    if abs(s["wind"]) > 5:
        if action.decision == "stabilize":
            score += 0.3

    # Priority 3 — Near-zero velocity window → lock it in
    if abs(s["velocity"]) < 5:
        if action.decision == "maintain":
            score += 0.2

    return float(max(0.0, min(score, 1.0)))


# ────────────────────────────────────────────────────────────────────────────
# Task registry — enumerate all tasks for the automated checker
# ────────────────────────────────────────────────────────────────────────────
TASKS = {
    "easy":   task_easy,
    "medium": task_medium,
    "hard":   task_hard,
}