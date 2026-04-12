---
title: Rocket Landing OpenEnv
emoji: 🚀
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
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

| Task   | Difficulty | Description                                                           | Score range |
|--------|------------|-----------------------------------------------------------------------|-------------|
| easy   | Easy       | Agent picks a valid, contextually appropriate action                  | 0.01 – 0.99 |
| medium | Medium     | Agent applies correct height-aware and velocity-aware thrust strategy | 0.01 – 0.99 |
| hard   | Hard       | Agent handles engine failure + wind + low altitude simultaneously     | 0.01 – 0.99 |

---

## API Endpoints

| Method | Path                | Description                              |
|--------|---------------------|------------------------------------------|
| GET    | `/health`           | Health check → `{"status":"healthy"}`    |
| GET    | `/metadata`         | Environment metadata with task manifest  |
| GET    | `/schema`           | Action/observation schemas + tasks       |
| GET    | `/tasks`            | List all tasks with grader strings       |
| POST   | `/reset`            | Reset env, returns Observation           |
| POST   | `/step`             | Advance one step with Action             |
| GET    | `/state`            | Get raw current state dict               |
| POST   | `/mcp`              | MCP JSON-RPC endpoint                    |
| POST   | `/tasks/{name}/run` | Run a full deterministic episode         |

---

## Environment Variables

| Variable       | Description                              | Default                           |
|----------------|------------------------------------------|-----------------------------------|
| `API_BASE_URL` | OpenAI-compatible API base URL           | `https://openrouter.ai/api/v1`   |
| `MODEL_NAME`   | Model identifier for inference           | `openai/gpt-4o-mini`             |
| `HF_TOKEN`     | Hugging Face / API key (bearer token)    | *(must be set)*                   |

---

## Baseline Scores

| Task   | Baseline agent score |
|--------|---------------------|
| easy   | ~0.85               |
| medium | ~0.75               |
| hard   | ~0.60               |

---

## Project Structure

```
rocket-landing-openenv/
├── environment.py    # Core environment
├── tasks.py          # Task graders (easy / medium / hard)
├── inference.py      # Baseline inference script
├── openenv.yaml      # OpenEnv specification manifest
├── requirements.txt
├── Dockerfile
├── README.md
└── server/
    └── app.py        # FastAPI server
```

---

## Team

- Jainam Dedhia (Team Lead)
- Maher Dhami
- Tirth Shah