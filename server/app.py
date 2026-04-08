"""
app.py — FastAPI server exposing the OpenEnv HTTP API.
"""

import sys
import os

# Ensure /app root is on the path regardless of how uvicorn is invoked
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from environment import RocketLandingEnv, Action, Observation, Reward
from tasks import TASKS,GRADERS, run_task_episode

app = FastAPI(
    title="Rocket Landing OpenEnv",
    description="OpenEnv-compliant rocket landing environment",
    version="2.0.0",
)

env = RocketLandingEnv()


@app.get("/")
def health():
    return {"status": "ok", "env": "rocket-landing-env", "version": "2.0.0"}


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


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "name":        name,
                "description": meta["description"],
                "grader":      meta["grader_name"],
                "module":      meta["module"],
                "score_range": [0.0, 1.0],
                "endpoint":    f"/tasks/{name}/run",
            }
            for name, meta in TASKS.items()
        ]
    }


@app.get("/tasks/{task_name}")
def get_task(task_name: str):
    if task_name not in TASKS:
        raise HTTPException(
            status_code=404,
            detail=f"Task '{task_name}' not found. Available: {list(TASKS.keys())}",
        )
    meta = TASKS[task_name]
    return {
        "name":        task_name,
        "description": meta["description"],
        "grader":      meta["grader_name"],
        "module":      meta["module"],
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
            "task":        task_name,
            "score":       score,
            "score_range": [0.0, 1.0],
            "grader":      TASKS[task_name]["grader_name"],
            "module":      TASKS[task_name]["module"],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=True)

if __name__ == "__main__":
    main()