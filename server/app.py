"""
server/app.py — Rocket Landing OpenEnv server.
DO NOT use create_app() — it registers /metadata and /schema with its own
response models that the hackathon validator does not recognise.
We build a plain FastAPI app manually instead.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from environment import RocketLandingEnv, Action as EnvAction, Reward
from tasks import TASKS, run_task_episode


# ── Pydantic request/response models ─────────────────────────────────────────

class StepRequest(BaseModel):
    decision: str = "maintain"


class ResetResponse(BaseModel):
    height: float
    velocity: float
    fuel: float
    engine_status: str
    wind: float
    step: int
    max_steps: int
    last_action: str | None = None


# ── Shared env instance ───────────────────────────────────────────────────────

_env = RocketLandingEnv()


# ── Task manifest — single source of truth, same in /tasks /schema /metadata ─

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


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Rocket Landing OpenEnv",
    version="1.0.0",
    description="Autonomous rocket landing environment for OpenEnv hackathon.",
)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "healthy"}


# ── Metadata ──────────────────────────────────────────────────────────────────

@app.get("/metadata")
def metadata():
    return {
        "name": "rocket-landing-env",
        "version": "1.0.0",
        "description": (
            "Autonomous rocket landing environment where an LLM agent "
            "controls thrust to safely land a descending rocket."
        ),
        "tasks": TASK_MANIFEST,
    }


# ── Schema ────────────────────────────────────────────────────────────────────

@app.get("/schema")
def schema():
    return {
        "action": {
            "type": "object",
            "properties": {
                "decision": {
                    "type": "string",
                    "enum": RocketLandingEnv.VALID_ACTIONS,
                }
            },
            "required": ["decision"],
        },
        "observation": {
            "type": "object",
            "properties": {
                "height": {"type": "number"},
                "velocity": {"type": "number"},
                "fuel": {"type": "number"},
                "engine_status": {"type": "string"},
                "wind": {"type": "number"},
                "step": {"type": "integer"},
                "max_steps": {"type": "integer"},
                "last_action": {"type": ["string", "null"]},
            },
        },
        "state": {"type": "object"},
        "tasks": TASK_MANIFEST,
    }


# ── Tasks ─────────────────────────────────────────────────────────────────────

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


# ── Reset ─────────────────────────────────────────────────────────────────────

@app.post("/reset")
def reset():
    obs = _env.reset()
    return obs.model_dump()


# ── Step ──────────────────────────────────────────────────────────────────────

@app.post("/step")
def step(request: StepRequest):
    try:
        action = EnvAction(decision=request.decision)
        obs, reward, done, info = _env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": reward.score,
            "done": done,
            "info": info,
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ── State ─────────────────────────────────────────────────────────────────────

@app.get("/state")
def state():
    try:
        return _env.state()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ── MCP (required by openenv spec) ───────────────────────────────────────────

@app.post("/mcp")
def mcp():
    return {"jsonrpc": "2.0", "result": {"status": "ok"}, "id": None}


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()