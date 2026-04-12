"""
app.py — Root FastAPI app for Rocket Landing OpenEnv.
Mirrors Paarth's structure exactly: plain FastAPI, no create_app().
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from environment import RocketLandingEnv, Action as EnvAction
from tasks import TASKS, run_task_episode

app = FastAPI(title="Rocket Landing OpenEnv", version="1.0.0")

_env = RocketLandingEnv()

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


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/metadata")
def metadata():
    return {
        "name": "rocket-landing-env",
        "version": "1.0.0",
        "description": "Autonomous rocket landing environment where an LLM agent controls thrust to safely land a descending rocket.",
        "tasks": TASK_MANIFEST,
    }


@app.get("/schema")
def schema():
    return {
        "action": {
            "type": "object",
            "properties": {"decision": {"type": "string", "enum": RocketLandingEnv.VALID_ACTIONS}},
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


@app.get("/tasks")
def list_tasks():
    return TASK_MANIFEST


@app.post("/tasks/{task_name}/run")
def run_task(task_name: str):
    if task_name not in TASKS:
        raise HTTPException(status_code=404, detail=f"Task '{task_name}' not found.")
    try:
        score = run_task_episode(task_name)
        return {"score": score, "done": True}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class StepRequest(BaseModel):
    decision: str = "maintain"


@app.post("/reset")
def reset():
    obs = _env.reset()
    return obs.model_dump()


@app.post("/step")
def step(request: StepRequest):
    try:
        action = EnvAction(decision=request.decision)
        obs, reward, done, info = _env.step(action)
        return {"observation": obs.model_dump(), "reward": reward.score, "done": done, "info": info}
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/state")
def state():
    try:
        return _env.state()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/mcp")
def mcp():
    return {"jsonrpc": "2.0", "result": {"status": "ok"}, "id": None}