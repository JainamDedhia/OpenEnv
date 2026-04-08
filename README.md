---
title: Rocket Landing OpenEnv
emoji: 🚀
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
tags:
  - openenv
---

# 🚀 Rocket Landing OpenEnv

An **OpenEnv-compliant** reinforcement learning environment where an AI agent
must autonomously land a descending rocket safely — managing thrust, wind
compensation, and engine-failure emergencies within a fixed step budget.

---

## Environment Description

The rocket begins at a random altitude (50–80 m) with a random downward
velocity (-12 to -6 m/s). Each step, the agent chooses one action; the physics
simulator updates height, velocity, fuel, and wind. A stochastic engine failure
can occur with 15% probability per step (resolved only by `emergency_burn`).
The episode terminates after **15 steps** or when the rocket reaches the ground.

**Safe landing condition:** `height ≤ 2 m` AND `-3 ≤ velocity ≤ 0.5 m/s`

This environment models the **autonomous control decision-making** problem that
real aerospace systems (SpaceX Falcon 9, Rocket Lab Electron) solve at each
timestep: given sensor readings, what thrust command minimises landing error?

---

## Observation Space

| Field           | Type    | Description                                |
|-----------------|---------|--------------------------------------------|
| `height`        | float   | Rocket altitude in metres                  |
| `velocity`      | float   | Vertical velocity (negative = descending)  |
| `fuel`          | float   | Remaining fuel fraction [0, 1]             |
| `engine_status` | string  | `"normal"` or `"failure"`                  |
| `wind`          | float   | Lateral wind speed in m/s                  |
| `step`          | int     | Current step index                         |
| `max_steps`     | int     | Total steps allowed per episode (15)       |
| `last_action`   | string? | Last action taken (null at reset)          |

## Action Space

| Action            | Effect                                         |
|-------------------|------------------------------------------------|
| `increase_thrust` | +3.0 m/s² upward (braking)                    |
| `decrease_thrust` | -1.0 m/s² (allow faster descent)              |
| `maintain`        | No thrust change                               |
| `stabilize`       | Counter wind: thrust = max(0, -wind×0.3)+0.5  |
| `emergency_burn`  | Emergency +5.0 m/s² upward, clears failure    |

---

## Tasks & Graders

| Task   | Difficulty | What is tested                                                      | Score range |
|--------|------------|---------------------------------------------------------------------|-------------|
| easy   | Easy       | Agent picks a valid, contextually appropriate action                | 0.0 – 1.0  |
| medium | Medium     | Agent applies correct height-aware and velocity-aware thrust strategy | 0.0 – 1.0 |
| hard   | Hard       | Agent handles engine failure + wind + low altitude simultaneously    | 0.0 – 1.0 |

All graders return graduated scores (not binary) to reward partial progress.

---

## API Endpoints

| Method | Path              | Description                              |
|--------|-------------------|------------------------------------------|
| GET    | `/`               | Health check → `{"status":"ok"}`         |
| POST   | `/reset`          | Reset env, returns Observation           |
| POST   | `/step`           | Advance one step with Action             |
| GET    | `/state`          | Get raw current state dict               |
| GET    | `/tasks`          | List all tasks with descriptions         |
| POST   | `/tasks/{name}/run` | Run a full deterministic episode       |

---

## Setup & Running Locally

### Prerequisites
- Python 3.11+
- Docker (for containerised deployment)

### Install

```bash
pip install -r requirements.txt
```

### Run the API server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Run the inference script

```bash
export API_BASE_URL="https://openrouter.ai/api/v1"
export MODEL_NAME="openai/gpt-4o-mini"
export HF_TOKEN="your-api-key-here"

python inference.py
```

Expected stdout:
```
[START] task=rocket-landing env=rocket-landing-openenv model=openai/gpt-4o-mini
[STEP] step=1 action=increase_thrust reward=0.58 done=false error=null
[STEP] step=2 action=maintain reward=0.61 done=false error=null
...
[END] success=true steps=15 score=0.6200 rewards=0.58,0.61,...
```

### Docker

```bash
docker build -t rocket-landing-env .
docker run -p 7860:7860 \
  -e HF_TOKEN="your-key" \
  -e MODEL_NAME="openai/gpt-4o-mini" \
  -e API_BASE_URL="https://openrouter.ai/api/v1" \
  rocket-landing-env
```

---

## Environment Variables

| Variable      | Description                              | Default                           |
|---------------|------------------------------------------|-----------------------------------|
| `API_BASE_URL`| OpenAI-compatible API base URL           | `https://openrouter.ai/api/v1`   |
| `MODEL_NAME`  | Model identifier for inference           | `openai/gpt-4o-mini`             |
| `HF_TOKEN`    | Hugging Face / API key (bearer token)    | *(must be set)*                   |

---

## Baseline Scores

| Task   | Baseline agent score | Notes                              |
|--------|---------------------|------------------------------------|
| easy   | ~0.85               | Rule-based rotation, always valid  |
| medium | ~0.75               | Height-aware braking agent         |
| hard   | ~0.60               | Full situational-awareness agent   |

---

## Resource Requirements

- vCPU: 2
- Memory: 8 GB
- Inference runtime: < 20 minutes

---

## Project Structure

```
rocket-landing-openenv/
├── app.py            ← symlink / alias (optional, for flat imports)
├── environment.py    # Core environment (reset/step/state + Pydantic models)
├── tasks.py          # Task graders (easy / medium / hard)
├── inference.py      # Baseline inference script (root, as required)
├── openenv.yaml      # OpenEnv specification manifest
├── requirements.txt
├── Dockerfile
├── README.md
└── server/
    └── app.py        # FastAPI server (OpenEnv HTTP API)
```

---

## Reward Design

Per-step reward is always in **[0.0, 1.0]**, computed as:

| Component         | Weight | Signal                                             |
|-------------------|--------|----------------------------------------------------|
| Altitude progress | 40%    | Lower altitude → higher score (approach to ground) |
| Velocity quality  | 30%    | Slow, controlled descent → higher score            |
| Fuel conservation | 15%    | More remaining fuel → higher score                 |
| Landing bonus     | 15%    | Perfect landing (h≤2m, v∈[-3,0.5]) → 1.0 bonus   |

This provides dense, non-sparse reward signal across the full trajectory.

---

## Team

- Jainam Dedhia (Team Lead)
- Maher Dhami
- Tirth Shah