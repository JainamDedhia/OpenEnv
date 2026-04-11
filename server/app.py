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


# ── Required OpenEnv compliance endpoints ─────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/metadata")
def metadata():
    return {
        "name": "rocket-landing-env",
        "description": (
            "Autonomous rocket landing environment where an LLM agent controls "
            "thrust to safely land a descending rocket within a fixed step budget, "
            "under stochastic wind and engine-failure conditions."
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
                "height":        {"type": "number",  "description": "Altitude in metres"},
                "velocity":      {"type": "number",  "description": "Vertical velocity (negative = descending)"},
                "fuel":          {"type": "number",  "description": "Remaining fuel fraction [0, 1]"},
                "engine_status": {"type": "string",  "description": "normal or failure"},
                "wind":          {"type": "number",  "description": "Lateral wind speed in m/s"},
                "step":          {"type": "integer", "description": "Current step index"},
                "max_steps":     {"type": "integer", "description": "Total steps allowed"},
                "last_action":   {"type": "string",  "description": "Last action taken"},
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


# ── Original health / root ────────────────────────────────────────────────────

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
            "grader": {
                "type": "llm",
                "prompt_template": "Score the rocket landing 0.0 to 1.0 based on whether the agent selected a valid and contextually appropriate action given the current state.",
            },
        },
        {
            "id":          "medium",
            "difficulty":  "medium",
            "max_steps":   15,
            "description": "Agent applies height-aware and velocity-aware thrust decisions.",
            "has_grader":  True,
            "grader": {
                "type": "llm",
                "prompt_template": "Score the rocket landing 0.0 to 1.0 based on whether the agent applied correct height-aware and velocity-aware thrust strategy.",
            },
        },
        {
            "id":          "hard",
            "difficulty":  "hard",
            "max_steps":   15,
            "description": "Agent handles engine failure, high wind, low altitude, and velocity management simultaneously.",
            "has_grader":  True,
            "grader": {
                "type": "llm",
                "prompt_template": "Score the rocket landing 0.0 to 1.0 based on how well the agent handled engine failure, high wind, and low altitude simultaneously.",
            },
        },
    ]


@app.get("/tasks/{task_name}")
def get_task(task_name: str):
    if task_name not in TASKS:
        raise HTTPException(
            status_code=404,
            detail=f"Task '{task_name}' not found. Available: {list(TASKS.keys())}",
        )
    meta = TASKS[task_name]
    return {
        "id":          task_name,
        "difficulty":  meta.get("difficulty", task_name),
        "description": meta["description"],
        "has_grader":  True,
        "grader": {
            "type": "llm",
            "prompt_template": f"Score the rocket landing 0.0 to 1.0 for task: {task_name}",
        },
        "score_range": [0.0, 1.0],
        "endpoint":    f"/tasks/{task_name}/run",
    }


@app.post("/tasks/{task_name}/run")
def run_task(task_name: str):
    if task_name not in TASKS:
        raise HTTPException(
            status_code=404,
            detail=f"Task '{task_name}' not found. Available: {list(TASKS.keys())}",
        )
    try:
        score = run_task_episode(task_name)
        return {
            "score": score,
            "done":  True,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=True)


if __name__ == "__main__":
    main()