"""
app.py — FastAPI server exposing the OpenEnv HTTP API.

Endpoints:
    GET  /           health check → 200
    POST /reset      reset the environment
    POST /step       advance one step
    GET  /state      get current raw state
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from environment import RocketLandingEnv, Action, Observation, Reward
from pydantic import BaseModel

app = FastAPI(
    title="Rocket Landing OpenEnv",
    description="OpenEnv-compliant rocket landing environment API",
    version="1.0.0",
)

# Single shared environment instance (stateless per-session in production
# would use session IDs, but single-instance is fine for the hackathon eval)
env = RocketLandingEnv()


# ──────────────────────────────────────────────────────────
# Health check (automated ping by the checker)
# ──────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "ok", "env": "rocket-landing-env", "version": "1.0.0"}


# ──────────────────────────────────────────────────────────
# POST /reset
# ──────────────────────────────────────────────────────────
@app.post("/reset", response_model=Observation)
def reset():
    obs = env.reset()
    return obs


# ──────────────────────────────────────────────────────────
# POST /step
# ──────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────
# GET /state
# ──────────────────────────────────────────────────────────
@app.get("/state")
def state():
    try:
        return JSONResponse(content=env.state())
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ──────────────────────────────────────────────────────────
# Local dev entry point
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)