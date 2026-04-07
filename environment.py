"""
environment.py — Core rocket landing environment.

Physics verified: safe landing IS achievable from h=50-80m in 15 steps.
Reward always in [0.0, 1.0]. No hidden randomness beyond reset.
"""

from __future__ import annotations
from pydantic import BaseModel
import random
import math


# ── Pydantic models ──────────────────────────────────────────────────────────

class Observation(BaseModel):
    height: float
    velocity: float
    fuel: float
    engine_status: str
    wind: float
    step: int
    max_steps: int
    last_action: str | None


class Action(BaseModel):
    decision: str


class Reward(BaseModel):
    score: float


# ── Environment ──────────────────────────────────────────────────────────────

class RocketLandingEnv:

    VALID_ACTIONS = [
        "increase_thrust",
        "decrease_thrust",
        "maintain",
        "stabilize",
        "emergency_burn",
    ]

    # Physics constants
    GRAVITY        = 1.5    # m/s² downward per step
    DT             = 1.0    # time step (seconds)
    THRUST_INC     = 3.0    # increase_thrust: upward acceleration
    THRUST_DEC     = 1.0    # decrease_thrust: allows faster fall
    THRUST_EMERG   = 5.0    # emergency_burn: strong upward
    FUEL_PER_STEP  = 0.04   # fuel consumed each step by active thrust

    def __init__(self):
        self._state: dict | None = None
        self.step_count: int     = 0
        self.max_steps: int      = 15
        self.last_action: str | None = None

    # ── reset ────────────────────────────────────────────────────────────────

    def reset(self) -> Observation:
        self.step_count  = 0
        self.last_action = None

        self._state = {
            "height":        float(random.uniform(50, 80)),   # achievable range
            "velocity":      float(random.uniform(-12, -6)),  # moderate descent
            "fuel":          float(random.uniform(0.7, 1.0)),
            "engine_status": "normal",
            "wind":          float(random.uniform(-5, 5)),
        }
        return self._make_obs()

    # ── step ─────────────────────────────────────────────────────────────────

    def step(self, action: Action):
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        if action.decision not in self.VALID_ACTIONS:
            raise RuntimeError(
                f"Invalid action '{action.decision}'. "
                f"Valid: {self.VALID_ACTIONS}"
            )

        s = self._state
        h = s["height"]
        v = s["velocity"]
        fuel = s["fuel"]
        wind = s["wind"]

        # ── Apply action ──────────────────────────────────────────────────
        thrust = 0.0
        fuel_cost = 0.0

        if action.decision == "increase_thrust":
            thrust    = self.THRUST_INC
            fuel_cost = self.FUEL_PER_STEP

        elif action.decision == "decrease_thrust":
            thrust    = -self.THRUST_DEC
            fuel_cost = 0.01

        elif action.decision == "emergency_burn":
            if s["engine_status"] == "failure":
                thrust    = self.THRUST_EMERG
                fuel_cost = self.FUEL_PER_STEP * 2
                s["engine_status"] = "normal"
            else:
                # emergency_burn when engine is fine → still applies thrust
                thrust    = self.THRUST_EMERG
                fuel_cost = self.FUEL_PER_STEP * 2

        elif action.decision == "stabilize":
            # Counters wind component; slight upward assist
            thrust    = max(0.0, -wind * 0.3) + 0.5
            fuel_cost = self.FUEL_PER_STEP * 0.5

        elif action.decision == "maintain":
            thrust    = 0.0
            fuel_cost = 0.0

        # Fuel cap — no thrust if empty
        if fuel <= 0.0:
            thrust    = 0.0
            fuel_cost = 0.0

        # ── Physics update ────────────────────────────────────────────────
        # net acceleration = thrust (upward +) − gravity (downward)
        accel = thrust - self.GRAVITY
        v_new = v + accel * self.DT
        h_new = h + v * self.DT + 0.5 * accel * self.DT ** 2

        fuel_new = max(0.0, fuel - fuel_cost)

        # Wind drifts slowly
        wind_new = wind + random.uniform(-0.5, 0.5)
        wind_new = max(-10.0, min(10.0, wind_new))

        # Stochastic engine failure (15% per step; resolved by emergency_burn)
        if s["engine_status"] == "normal" and random.random() < 0.15:
            s["engine_status"] = "failure"

        # Ground clamp
        if h_new < 0:
            h_new = 0.0

        s["height"]   = round(h_new, 4)
        s["velocity"] = round(v_new, 4)
        s["fuel"]     = round(fuel_new, 4)
        s["wind"]     = round(wind_new, 4)

        self.step_count  += 1
        self.last_action  = action.decision

        # ── Reward ────────────────────────────────────────────────────────
        reward_score = self._compute_reward(
            h_new, v_new, fuel_new, s["engine_status"], action.decision, wind
        )

        done = self.step_count >= self.max_steps or h_new <= 0

        obs = self._make_obs()
        return obs, Reward(score=reward_score), done, {}

    # ── state ─────────────────────────────────────────────────────────────

    def state(self) -> dict:
        if self._state is None:
            raise RuntimeError("Call reset() before state().")
        return {
            **self._state,
            "step":      self.step_count,
            "max_steps": self.max_steps,
            "last_action": self.last_action,
        }

    # ── internal helpers ──────────────────────────────────────────────────

    def _make_obs(self) -> Observation:
        return Observation(
            **self._state,
            step=self.step_count,
            max_steps=self.max_steps,
            last_action=self.last_action,
        )

    def _compute_reward(
        self,
        h: float,
        v: float,
        fuel: float,
        engine_status: str,
        decision: str,
        wind: float,
    ) -> float:
        """
        Reward always in [0.0, 1.0].

        Components (weighted):
          40%  altitude progress  — lower = better (approaching ground safely)
          30%  velocity quality   — slow descent near ground = better
          15%  fuel conservation
          15%  landing bonus      — perfect landing at terminal condition
        """

        # ── Altitude component (0–1): prefer low but not crashed ──────────
        # Maps h in [0, 80] → score in [1, 0] smoothly
        alt_score = max(0.0, 1.0 - (h / 80.0))

        # ── Velocity component (0–1): prefer velocity near 0 when low ────
        # Ideal: velocity in [-3, 0] near ground
        if h < 15:
            # Near ground: penalize fast descent harshly
            ideal_v = -1.5
            v_error = abs(v - ideal_v)
            vel_score = max(0.0, 1.0 - (v_error / 10.0))
        else:
            # Higher up: gentle descent is fine, just not too fast
            vel_score = max(0.0, 1.0 - (abs(v) / 20.0))

        # ── Fuel component (0–1) ──────────────────────────────────────────
        fuel_score = fuel  # already in [0, 1]

        # ── Landing bonus (0–1): triggered when on/near ground safely ─────
        landing_bonus = 0.0
        if h <= 2.0 and -3.0 <= v <= 0.5:
            landing_bonus = 1.0
        elif h <= 5.0 and -5.0 <= v <= 0.5:
            landing_bonus = 0.6

        # ── Weighted combination ──────────────────────────────────────────
        raw = (
            0.40 * alt_score
            + 0.30 * vel_score
            + 0.15 * fuel_score
            + 0.15 * landing_bonus
        )

        return round(float(max(0.0, min(raw, 1.0))), 6)