"""
server/app.py — OpenEnv-compliant FastAPI server for Rocket Landing environment.
Uses openenv-core's create_app — auto-registers /health /metadata /schema /mcp /ws /reset /step /state
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from openenv.core.env_server.types import Action as OpenEnvAction
from openenv.core.env_server.types import Observation as OpenEnvObservation

from environment import RocketLandingEnv, Action, Reward
from tasks import TASKS, run_task_episode

from fastapi import HTTPException


# ── Pydantic models for openenv-core ─────────────────────────────────────────

class RocketAction(OpenEnvAction):
    decision: str = "maintain"


class RocketObservation(OpenEnvObservation):
    height: float = 0.0
    velocity: float = 0.0
    fuel: float = 1.0
    engine_status: str = "normal"
    wind: float = 0.0
    step: int = 0
    max_steps: int = 15
    last_action: str | None = None
    reward: float | None = None
    done: bool = False


# ── Environment wrapper ───────────────────────────────────────────────────────

class RocketEnvWrapper(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._env = RocketLandingEnv()

    def reset(self, **kwargs) -> RocketObservation:
        obs = self._env.reset()
        return RocketObservation(
            height=obs.height,
            velocity=obs.velocity,
            fuel=obs.fuel,
            engine_status=obs.engine_status,
            wind=obs.wind,
            step=obs.step,
            max_steps=obs.max_steps,
            last_action=obs.last_action,
            reward=None,
            done=False,
        )

    def step(self, action: RocketAction) -> RocketObservation:
        env_action = Action(decision=action.decision)
        obs, reward, done, _ = self._env.step(env_action)
        return RocketObservation(
            height=obs.height,
            velocity=obs.velocity,
            fuel=obs.fuel,
            engine_status=obs.engine_status,
            wind=obs.wind,
            step=obs.step,
            max_steps=obs.max_steps,
            last_action=obs.last_action,
            reward=reward.score,
            done=done,
        )

    @property
    def state(self) -> State:
        try:
            s = self._env.state()
        except RuntimeError:
            s = {}
        return State(
            episode_id="rocket-landing",
            step_count=s.get("step", 0),
        )


# ── Central task manifest — single source of truth, mirrors openenv.yaml ─────

TASK_MANIFEST = [
    {
        "id": "easy",
        "difficulty": "easy",
        "max_steps": 15,
        "description": "Agent selects a valid and contextually appropriate action.",
        "grader": "tasks:task_easy",
        "reward_range": [0.01, 0.99],
    },
    {
        "id": "medium",
        "difficulty": "medium",
        "max_steps": 15,
        "description": "Agent applies height-aware and velocity-aware thrust decisions.",
        "grader": "tasks:task_medium",
        "reward_range": [0.01, 0.99],
    },
    {
        "id": "hard",
        "difficulty": "hard",
        "max_steps": 15,
        "description": "Agent handles engine failure, wind, low altitude simultaneously.",
        "grader": "tasks:task_hard",
        "reward_range": [0.01, 0.99],
    },
]


# ── Create app via openenv-core ───────────────────────────────────────────────

app = create_app(
    RocketEnvWrapper,
    RocketAction,
    RocketObservation,
    env_name="rocket-landing-env",
    max_concurrent_envs=4,
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/metadata")
def metadata():
    return {
        "name": "rocket-landing-env",
        "version": "1.0.0",
        "runtime": "fastapi",
        "entrypoint": "environment:RocketLandingEnv",
        "app": "server.app:app",
        "description": (
            "Autonomous rocket landing environment where an LLM agent "
            "controls thrust to safely land a descending rocket."
        ),
        "tasks": TASK_MANIFEST,
    }


@app.get("/schema")
def schema():
    return {
        "name": "rocket-landing-env",
        "version": "1.0.0",
        "action": RocketAction.model_json_schema(),
        "observation": RocketObservation.model_json_schema(),
        "state": {"type": "object"},
        "tasks": TASK_MANIFEST,
    }


@app.get("/tasks")
def list_tasks():
    return TASK_MANIFEST


@app.post("/tasks/{task_name}/run")
def run_task(task_name: str):
    if task_name not in TASKS:
        raise HTTPException(
            status_code=404, detail=f"Task '{task_name}' not found."
        )
    try:
        score = run_task_episode(task_name)
        return {"score": score, "done": True}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()