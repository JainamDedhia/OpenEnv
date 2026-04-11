"""
tasks.py — Task definitions and episode runners for Rocket Landing OpenEnv.
"""

from __future__ import annotations
from environment import RocketLandingEnv, Action


# ── Rule-based agents ────────────────────────────────────────────────────────

def _rule_agent_easy(obs: dict) -> str:
    step = obs["step"] % 3
    actions = ["increase_thrust", "maintain", "stabilize"]
    return actions[step]


def _rule_agent_medium(obs: dict) -> str:
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


def _rule_agent_fuel(obs: dict) -> str:
    fuel = obs["fuel"]
    v    = obs["velocity"]
    if fuel > 0.5:
        return "increase_thrust"
    if v < -3:
        return "maintain"
    return "decrease_thrust"


def _rule_agent_wind(obs: dict) -> str:
    wind   = obs["wind"]
    engine = obs["engine_status"]
    if engine == "failure":
        return "emergency_burn"
    if abs(wind) > 4:
        return "stabilize"
    return "increase_thrust"


def _rule_agent_precision(obs: dict) -> str:
    h = obs["height"]
    v = obs["velocity"]
    engine = obs["engine_status"]
    if engine == "failure":
        return "emergency_burn"
    if h < 10:
        if v < -2:
            return "increase_thrust"
        return "maintain"
    if h < 30 and v < -4:
        return "increase_thrust"
    return "maintain"


# ── Episode runner ────────────────────────────────────────────────────────────

def _run_episode(agent_fn, seed_height: float, seed_velocity: float) -> float:
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


# ── Graders ──────────────────────────────────────────────────────────────────

def task_easy(env: RocketLandingEnv, action: Action) -> float:
    if action.decision not in RocketLandingEnv.VALID_ACTIONS:
        return 0.0
    s      = env.state()
    h      = s["height"]
    v      = s["velocity"]
    engine = s["engine_status"]
    if engine == "failure":
        if action.decision == "emergency_burn":
            return 1.0
        if action.decision in ("increase_thrust", "stabilize"):
            return 0.7
        return 0.4
    if h < 30 and v < -4:
        if action.decision in ("increase_thrust", "emergency_burn"):
            return 1.0
        if action.decision == "stabilize":
            return 0.7
        return 0.4
    if h >= 50:
        if action.decision in ("maintain", "stabilize", "increase_thrust"):
            return 1.0
        return 0.7
    if action.decision in ("increase_thrust", "maintain", "stabilize"):
        return 1.0
    return 0.7


def task_medium(env: RocketLandingEnv, action: Action) -> float:
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
    s = env.state()
    if s["engine_status"] == "failure":
        if action.decision == "emergency_burn":
            return 1.0
        if action.decision == "increase_thrust":
            return 0.4
        return 0.1
    score = 0.0
    if abs(s["wind"]) > 6:
        if action.decision == "stabilize":
            score += 0.35
        elif action.decision in ("maintain", "increase_thrust"):
            score += 0.15
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
    if abs(s["velocity"]) <= 3:
        if action.decision == "maintain":
            score += 0.25
        elif action.decision == "stabilize":
            score += 0.15
        else:
            score += 0.05
    return round(float(max(0.0, min(score, 1.0))), 6)


def task_fuel_management(env: RocketLandingEnv, action: Action) -> float:
    """Score based on fuel-efficient landing decisions."""
    s    = env.state()
    fuel = s["fuel"]
    v    = s["velocity"]
    h    = s["height"]
    if s["engine_status"] == "failure":
        return 1.0 if action.decision == "emergency_burn" else 0.0
    if fuel < 0.2:
        if action.decision == "maintain":
            return 1.0
        if action.decision == "decrease_thrust":
            return 0.8
        return 0.2
    if h < 20 and v < -3:
        if action.decision == "increase_thrust":
            return 1.0
        return 0.3
    if action.decision in ("maintain", "stabilize"):
        return 0.9
    if action.decision == "increase_thrust":
        return 0.6
    return 0.4


def task_wind_compensation(env: RocketLandingEnv, action: Action) -> float:
    """Score based on wind-aware stabilization decisions."""
    s    = env.state()
    wind = s["wind"]
    h    = s["height"]
    v    = s["velocity"]
    if s["engine_status"] == "failure":
        return 1.0 if action.decision == "emergency_burn" else 0.0
    if abs(wind) > 7:
        if action.decision == "stabilize":
            return 1.0
        if action.decision == "maintain":
            return 0.4
        return 0.2
    if abs(wind) > 4:
        if action.decision in ("stabilize", "increase_thrust"):
            return 1.0
        if action.decision == "maintain":
            return 0.6
        return 0.3
    if h < 20 and v < -3:
        if action.decision == "increase_thrust":
            return 1.0
        return 0.5
    return 0.7


def task_precision_landing(env: RocketLandingEnv, action: Action) -> float:
    """Score based on precision near-ground velocity control."""
    s = env.state()
    h = s["height"]
    v = s["velocity"]
    if s["engine_status"] == "failure":
        return 1.0 if action.decision == "emergency_burn" else 0.0
    if h <= 10:
        if -2.0 <= v <= 0.5:
            if action.decision == "maintain":
                return 1.0
            return 0.5
        if v < -2.0:
            if action.decision == "increase_thrust":
                return 1.0
            return 0.2
    if h <= 25 and v < -4:
        if action.decision == "increase_thrust":
            return 1.0
        if action.decision == "maintain":
            return 0.4
        return 0.2
    if action.decision in ("maintain", "increase_thrust", "stabilize"):
        return 0.8
    return 0.5


# ── Full episode run ──────────────────────────────────────────────────────────

def run_task_episode(task_name: str) -> float:
    configs = {
        "easy":              (_rule_agent_easy,      65.0, -8.0),
        "medium":            (_rule_agent_medium,    60.0, -10.0),
        "hard":              (_rule_agent_hard,      55.0, -12.0),
        "fuel_management":   (_rule_agent_fuel,      60.0, -9.0),
        "wind_compensation": (_rule_agent_wind,      58.0, -8.0),
        "precision_landing": (_rule_agent_precision, 50.0, -11.0),
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
        "grader_name": "task_easy",
        "module":      "tasks",
        "difficulty":  "easy",
    },
    "medium": {
        "description": "Agent applies height-aware and velocity-aware thrust decisions.",
        "grader":      task_medium,
        "grader_name": "task_medium",
        "module":      "tasks",
        "difficulty":  "medium",
    },
    "hard": {
        "description": "Agent handles engine failure, high wind, low altitude, and velocity management simultaneously.",
        "grader":      task_hard,
        "grader_name": "task_hard",
        "module":      "tasks",
        "difficulty":  "hard",
    },
    "fuel_management": {
        "description": "Agent conserves fuel while maintaining safe descent velocity.",
        "grader":      task_fuel_management,
        "grader_name": "task_fuel_management",
        "module":      "tasks",
        "difficulty":  "medium",
    },
    "wind_compensation": {
        "description": "Agent compensates for high wind using stabilize and thrust decisions.",
        "grader":      task_wind_compensation,
        "grader_name": "task_wind_compensation",
        "module":      "tasks",
        "difficulty":  "medium",
    },
    "precision_landing": {
        "description": "Agent achieves precise near-ground velocity control for safe touchdown.",
        "grader":      task_precision_landing,
        "grader_name": "task_precision_landing",
        "module":      "tasks",
        "difficulty":  "hard",
    },
}

GRADERS = {
    "easy":              task_easy,
    "medium":            task_medium,
    "hard":              task_hard,
    "fuel_management":   task_fuel_management,
    "wind_compensation": task_wind_compensation,
    "precision_landing": task_precision_landing,
}