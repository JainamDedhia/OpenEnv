from pydantic import BaseModel
import random


# -------------------------------
# Models
# -------------------------------
class Observation(BaseModel):
    height: float
    velocity: float
    fuel: float
    engine_status: str
    wind: float
    step: int
    last_action: str | None


class Action(BaseModel):
    decision: str


class Reward(BaseModel):
    score: float


# -------------------------------
# Environment
# -------------------------------
class RocketLandingEnv:

    VALID_ACTIONS = [
        "increase_thrust",
        "decrease_thrust",
        "maintain",
        "stabilize",
        "emergency_burn",
    ]

    def __init__(self):
        self.state_data = None
        self.step_count = 0
        self.max_steps = 6
        self.last_action = None

    def reset(self) -> Observation:
        self.step_count = 0
        self.last_action = None

        self.state_data = {
            "height": random.uniform(120, 280),
            "velocity": random.uniform(-28, -6),
            "fuel": random.uniform(0.5, 1.0),
            "engine_status": "normal",
            "wind": random.uniform(-10, 10),
        }

        return Observation(
            **self.state_data,
            step=self.step_count,
            last_action=self.last_action,
        )

    def state(self) -> dict:
        """Return the current raw state dict (required by OpenEnv spec)."""
        if self.state_data is None:
            raise RuntimeError("Environment has not been reset yet. Call reset() first.")
        return {
            **self.state_data,
            "step": self.step_count,
            "last_action": self.last_action,
        }

    def step(self, action: Action):
        if self.state_data is None:
            raise RuntimeError("Environment has not been reset yet. Call reset() first.")

        h = self.state_data["height"]
        v = self.state_data["velocity"]
        wind = self.state_data["wind"]

        score = 0.0

        # -----------------------------------------------
        # Random engine emergency (25 % chance per step)
        # -----------------------------------------------
        if random.random() < 0.25:
            self.state_data["engine_status"] = "failure"

        # -----------------------------------------------
        # Decision scoring logic
        # -----------------------------------------------
        if self.state_data["engine_status"] == "failure":
            if action.decision == "emergency_burn":
                score += 0.5
            else:
                score -= 0.2

        if abs(wind) > 6 and action.decision == "stabilize":
            score += 0.3

        if h > 160 and action.decision == "decrease_thrust":
            score += 0.3

        if h < 60 and action.decision == "increase_thrust":
            score += 0.3

        if abs(v) < 5 and action.decision == "maintain":
            score += 0.2

        # -----------------------------------------------
        # Anti-repetition penalty
        # -----------------------------------------------
        if self.last_action == action.decision:
            score -= 0.5

        # -----------------------------------------------
        # Physics simulation
        # -----------------------------------------------
        if action.decision == "decrease_thrust":
            v -= 1.5
        elif action.decision == "increase_thrust":
            v += 2.0
        elif action.decision == "emergency_burn":
            v += 3.5
        elif action.decision == "stabilize":
            v += (-wind * 0.2)
        # "maintain" → no thrust change

        v -= 0.8                                  # gravity
        h += v * 0.5                              # position update

        self.state_data["height"] = h
        self.state_data["velocity"] = v
        self.state_data["fuel"] = max(0.0, self.state_data["fuel"] - 0.05)
        self.state_data["wind"] += random.uniform(-1.5, 1.5)
        # emergency_burn resolves the current failure; otherwise carry it forward
        if action.decision == "emergency_burn":
            self.state_data["engine_status"] = "normal"
        # (random failure injection at top of next step may set it again)

        self.step_count += 1
        self.last_action = action.decision

        done = self.step_count >= self.max_steps

        # -----------------------------------------------
        # Landing bonus (only evaluated at terminal step)
        # -----------------------------------------------
        if done:
            if h < 10 and -3 <= v <= 0:
                score += 1.0

        score = max(0.0, min(score, 1.0))

        obs = Observation(
            **self.state_data,
            step=self.step_count,
            last_action=self.last_action,
        )
        return obs, Reward(score=score), done, {}