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

    import random
    random.seed(42)
    obs = env.reset()

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
    Graduated score based on action validity and situational appropriateness.
    - Invalid action: 0.0
    - Valid but passive when thrust is needed: 0.4
    - Valid and contextually reasonable: 0.7
    - Valid and contextually optimal: 1.0
    """
    if action.decision not in RocketLandingEnv.VALID_ACTIONS:
        return 0.0

    s = env.state()
    h = s["height"]
    v = s["velocity"]
    engine = s["engine_status"]

    # Emergency situation — emergency_burn is optimal
    if engine == "failure":
        if action.decision == "emergency_burn":
            return 1.0
        if action.decision in ("increase_thrust", "stabilize"):
            return 0.7
        return 0.4

    # Descending fast at low altitude — needs braking
    if h < 30 and v < -4:
        if action.decision in ("increase_thrust", "emergency_burn"):
            return 1.0
        if action.decision == "stabilize":
            return 0.7
        if action.decision == "maintain":
            return 0.4
        return 0.4  # decrease_thrust is counterproductive

    # High altitude, gentle conditions — maintain or stabilize fine
    if h >= 50:
        if action.decision in ("maintain", "stabilize", "increase_thrust"):
            return 1.0
        if action.decision == "decrease_thrust":
            return 0.7
        return 0.7

    # Mid altitude — any valid action acceptable, thrust preferred
    if action.decision in ("increase_thrust", "maintain", "stabilize"):
        return 1.0
    return 0.7


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
    Each sub-situation awards a bounded partial score — total clamped to [0, 1].
    """
    s = env.state()

    # Engine failure is highest priority — score it independently
    if s["engine_status"] == "failure":
        if action.decision == "emergency_burn":
            return 1.0
        if action.decision == "increase_thrust":
            return 0.4
        return 0.1

    score = 0.0

    # Wind component (max 0.35)
    if abs(s["wind"]) > 6:
        if action.decision == "stabilize":
            score += 0.35
        elif action.decision in ("maintain", "increase_thrust"):
            score += 0.15

    # Altitude + velocity component (max 0.40)
    if s["height"] < 20 and s["velocity"] < -3:
        if action.decision == "increase_thrust":
            score += 0.40
        elif action.decision == "emergency_burn":
            score += 0.30
        elif action.decision == "maintain":
            score += 0.10
    elif s["height"] < 40 and s["velocity"] < -5:
        if action.decision == "increase_thrust":
            score += 0.30
        elif action.decision == "maintain":
            score += 0.10

    # Velocity stability component (max 0.25)
    if abs(s["velocity"]) <= 3:
        if action.decision == "maintain":
            score += 0.25
        elif action.decision == "stabilize":
            score += 0.15
        else:
            score += 0.05

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
        "description": "Agent selects a valid and contextually appropriate action from the action space.",
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

# ── Explicit grader export (for OpenEnv validator discovery) ──────────────────

GRADERS = {
    "task_easy":   task_easy,
    "task_medium": task_medium,
    "task_hard":   task_hard,
}