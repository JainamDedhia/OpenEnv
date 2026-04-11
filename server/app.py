"""
app.py — FastAPI server exposing the OpenEnv HTTP API.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from environment import RocketLandingEnv, Action, Observation, Reward
from tasks import TASKS, GRADERS, run_task_episode

app = FastAPI(
    title="Rocket Landing OpenEnv",
    description="OpenEnv-compliant rocket landing environment",
    version="2.0.0",
)

env = RocketLandingEnv()


# ── OpenEnv compliance endpoints ──────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/metadata")
def metadata():
    return {
        "name": "rocket-landing-env",
        "description": (
            "Autonomous rocket landing environment where an LLM agent controls "
            "thrust to safely land a descending rocket within a fixed step budget."
        ),
        "version": "2.0.0",
        "author": "Jainam Dedhia, Maher Dhami, Tirth Shah",
    }


@app.get("/schema")
def schema():
    return {
        "action": {
            "type": "object",
            "properties": {
                "decision": {
                    "type": "string",
                    "enum": [
                        "increase_thrust",
                        "decrease_thrust",
                        "maintain",
                        "stabilize",
                        "emergency_burn",
                    ],
                }
            },
            "required": ["decision"],
        },
        "observation": {
            "type": "object",
            "properties": {
                "height":        {"type": "number"},
                "velocity":      {"type": "number"},
                "fuel":          {"type": "number"},
                "engine_status": {"type": "string"},
                "wind":          {"type": "number"},
                "step":          {"type": "integer"},
                "max_steps":     {"type": "integer"},
                "last_action":   {"type": "string"},
            },
        },
        "state": {
            "type": "object",
            "properties": {
                "height":        {"type": "number"},
                "velocity":      {"type": "number"},
                "fuel":          {"type": "number"},
                "engine_status": {"type": "string"},
                "wind":          {"type": "number"},
                "step":          {"type": "integer"},
                "max_steps":     {"type": "integer"},
                "last_action":   {"type": "string"},
            },
        },
    }


@app.post("/mcp")
async def mcp(request: Request):
    return {"jsonrpc": "2.0", "id": None, "result": {}}


# ── Root ──────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "env": "rocket-landing-env", "version": "2.0.0"}


# ── Core OpenEnv endpoints ────────────────────────────────────────────────────

@app.post("/reset", response_model=Observation)
def reset():
    return env.reset()


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict


@app.post("/step", response_model=StepResponse)
def step(action: Action):
    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state")
def state():
    try:
        return JSONResponse(content=env.state())
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ── Task endpoints ────────────────────────────────────────────────────────────

@app.get("/tasks")
def list_tasks():
    return [
        {
            "id":          "easy",
            "difficulty":  "easy",
            "max_steps":   15,
            "description": "Agent selects a valid and contextually appropriate action from the action space.",
            "has_grader":  True,
            "grader": {"type": "llm", "prompt_template": "Score the rocket landing 0.0 to 1.0 based on action validity and context appropriateness."},
        },
        {
            "id":          "medium",
            "difficulty":  "medium",
            "max_steps":   15,
            "description": "Agent applies height-aware and velocity-aware thrust decisions.",
            "has_grader":  True,
            "grader": {"type": "llm", "prompt_template": "Score the rocket landing 0.0 to 1.0 based on height-aware and velocity-aware thrust strategy."},
        },
        {
            "id":          "hard",
            "difficulty":  "hard",
            "max_steps":   15,
            "description": "Agent handles engine failure, high wind, low altitude simultaneously.",
            "has_grader":  True,
            "grader": {"type": "llm", "prompt_template": "Score the rocket landing 0.0 to 1.0 based on handling engine failure, wind, and low altitude simultaneously."},
        },
        {
            "id":          "fuel_management",
            "difficulty":  "medium",
            "max_steps":   15,
            "description": "Agent conserves fuel while maintaining safe descent velocity.",
            "has_grader":  True,
            "grader": {"type": "llm", "prompt_template": "Score the rocket landing 0.0 to 1.0 based on fuel-efficient descent decisions."},
        },
        {
            "id":          "wind_compensation",
            "difficulty":  "medium",
            "max_steps":   15,
            "description": "Agent compensates for high wind using stabilize and thrust decisions.",
            "has_grader":  True,
            "grader": {"type": "llm", "prompt_template": "Score the rocket landing 0.0 to 1.0 based on wind compensation strategy."},
        },
        {
            "id":          "precision_landing",
            "difficulty":  "hard",
            "max_steps":   15,
            "description": "Agent achieves precise near-ground velocity control for safe touchdown.",
            "has_grader":  True,
            "grader": {"type": "llm", "prompt_template": "Score the rocket landing 0.0 to 1.0 based on precision near-ground velocity control."},
        },
    ]


@app.get("/tasks/{task_name}")
def get_task(task_name: str):
    if task_name not in TASKS:
        raise HTTPException(status_code=404, detail=f"Task '{task_name}' not found.")
    meta = TASKS[task_name]
    return {
        "id":          task_name,
        "difficulty":  meta.get("difficulty", task_name),
        "description": meta["description"],
        "has_grader":  True,
        "grader": {"type": "llm", "prompt_template": f"Score the rocket landing 0.0 to 1.0 for task: {task_name}"},
        "score_range": [0.0, 1.0],
    }


@app.post("/tasks/{task_name}/run")
def run_task(task_name: str):
    if task_name not in TASKS:
        raise HTTPException(status_code=404, detail=f"Task '{task_name}' not found.")
    try:
        score = run_task_episode(task_name)
        return {"score": score, "done": True}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=True)


if __name__ == "__main__":
    main()