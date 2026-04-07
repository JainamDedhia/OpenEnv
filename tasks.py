"""
tasks.py — Task definitions and episode runners for Rocket Landing OpenEnv.

Each task runs a full deterministic episode using a rule-based agent
and returns a score in [0.0, 1.0].
"""

from __future__ import annotations
from environment import RocketLandingEnv, Action


# ── Rule-based agents ────────────────────────────────────────────────────────

def _rule_agent_easy(obs: dict) -> str:
    """Always picks a valid action. Simple rotation strategy."""
    step = obs["step"] % 3
    actions = ["increase_thrust", "maintain", "stabilize"]
    return actions[step]


def _rule_agent_medium(obs: dict) -> str:
    """Height-aware braking agent. Brakes early to achieve safe landing."""
    h = obs["height"]
    v = obs["velocity"]
    engine = obs["engine_status"]

    if engine == "failure":
        return "emergency_burn"
    # Start braking when below 55m and velocity still fast
    if h < 55 and v < -3:
        return "increase_thrust"
    if -3 <= v <= 0:
        return "maintain"
    return "maintain"


def _rule_agent_hard(obs: dict) -> str:
    """
    Full situational-awareness agent:
    handles engine failure, wind, altitude, velocity.
    """
    engine = obs["engine_status"]
    wind   = obs["wind"]
    h      = obs["height"]
    v      = obs["velocity"]

    if engine == "failure":
        return "emergency_burn"
    if abs(wind) > 6:
        return "stabilize"
    if h < 55 and v < -3:
        return "increase_thrust"
    if -3 <= v <= 0:
        return "maintain"
    return "maintain"


# ── Episode runner ────────────────────────────────────────────────────────────

def _run_episode(agent_fn, seed_height: float, seed_velocity: float) -> float:
    """
    Run a complete deterministic episode with the given agent and
    fixed initial conditions. Returns the mean per-step reward.
    """
    env = RocketLandingEnv()

    # Manual seeded reset for determinism
    import random
    random.seed(42)
    obs = env.reset()

    # Override with deterministic starting state
    env._state["height"]        = seed_height
    env._state["velocity"]      = seed_velocity
    env._state["fuel"]          = 0.9
    env._state["engine_status"] = "normal"
    env._state["wind"]          = 2.0

    total_reward = 0.0
    steps        = 0
    done         = False

    while not done:
        obs_dict = env.state()
        decision = agent_fn(obs_dict)
        action   = Action(decision=decision)
        _, reward, done, _ = env.step(action)
        total_reward += reward.score
        steps        += 1

    return round(total_reward / max(steps, 1), 6)


# ── Graders (called via API) ──────────────────────────────────────────────────

def task_easy(env: RocketLandingEnv, action: Action) -> float:
    """
    Score 1.0 if action is valid, 0.0 otherwise.
    Tests basic action-space compliance.
    """
    return 1.0 if action.decision in RocketLandingEnv.VALID_ACTIONS else 0.0


def task_medium(env: RocketLandingEnv, action: Action) -> float:
    """
    Score based on height-appropriate thrust choice.
    """
    s = env.state()
    h = s["height"]
    v = s["velocity"]

    if s["engine_status"] == "failure":
        return 1.0 if action.decision == "emergency_burn" else 0.0

    if h > 40:
        if action.decision == "maintain":
            return 1.0
        if action.decision in ("decrease_thrust", "stabilize"):
            return 0.6
        return 0.2

    # Low altitude — need braking
    if v < -5:
        if action.decision == "increase_thrust":
            return 1.0
        if action.decision == "maintain":
            return 0.5
        return 0.1

    if action.decision == "maintain":
        return 1.0
    return 0.5


def task_hard(env: RocketLandingEnv, action: Action) -> float:
    """
    Multi-factor scoring: engine failure + wind + velocity management.
    """
    s = env.state()
    score = 0.0

    if s["engine_status"] == "failure":
        if action.decision == "emergency_burn":
            score += 0.5
        # else: no penalty, but no bonus
    else:
        if abs(s["wind"]) > 6:
            if action.decision == "stabilize":
                score += 0.3
            elif action.decision in ("maintain", "increase_thrust"):
                score += 0.15

        if s["height"] < 20 and s["velocity"] < -3:
            if action.decision == "increase_thrust":
                score += 0.3
            elif action.decision == "emergency_burn":
                score += 0.2

        if abs(s["velocity"]) <= 3:
            if action.decision == "maintain":
                score += 0.2
            else:
                score += 0.1

    return round(float(max(0.0, min(score, 1.0))), 6)


# ── Full episode run (called by /tasks/{name}/run) ────────────────────────────

def run_task_episode(task_name: str) -> float:
    """
    Run a full deterministic episode for the named task.
    Returns average reward in [0.0, 1.0].
    """
    configs = {
        "easy":   (_rule_agent_easy,   65.0, -8.0),
        "medium": (_rule_agent_medium, 60.0, -10.0),
        "hard":   (_rule_agent_hard,   55.0, -12.0),
    }
    if task_name not in configs:
        raise ValueError(f"Unknown task: {task_name}")

    agent_fn, h0, v0 = configs[task_name]
    return _run_episode(agent_fn, h0, v0)


# ── Task registry ─────────────────────────────────────────────────────────────

TASKS = {
    "easy": {
        "description": "Agent selects a valid action from the action space.",
        "grader":      task_easy,
    },
    "medium": {
        "description": "Agent applies height-aware and velocity-aware thrust decisions.",
        "grader":      task_medium,
    },
    "hard": {
        "description": (
            "Agent handles engine failure, high wind, low altitude, "
            "and velocity management simultaneously."
        ),
        "grader":      task_hard,
    },
}