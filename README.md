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

The rocket begins at a random altitude (120–280 m) with a random downward
velocity (-28 to -6 m/s). Each step, the agent chooses one action; the physics
simulator updates height, velocity, fuel, and wind. A stochastic engine failure
can occur with 25% probability per step. The episode terminates after 6 steps.

**Safe landing condition:** `height < 10 m` AND `-3 ≤ velocity ≤ 0 m/s`

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
| `last_action`   | string? | Last action taken (null at reset)          |

## Action Space

| Action            | Effect                                         |
|-------------------|------------------------------------------------|
| `increase_thrust` | +2.0 m/s² upward (braking)                    |
| `decrease_thrust` | -1.5 m/s² (allow faster descent)              |
| `maintain`        | No thrust change                               |
| `stabilize`       | Counter wind: `Δv = -wind × 0.2`              |
| `emergency_burn`  | Emergency +3.5 m/s² upward thrust             |

---

## Tasks & Graders

| Task   | Difficulty | What is tested                                              | Score range |
|--------|------------|-------------------------------------------------------------|-------------|
| easy   | Easy       | Agent picks a valid action from the action space            | 0.0 – 1.0  |
| medium | Medium     | Agent applies correct height-aware thrust strategy          | 0.0 – 1.0  |
| hard   | Hard       | Agent handles engine failure + wind + velocity simultaneously | 0.0 – 1.0 |

---

## API Endpoints

| Method | Path     | Description                     |
|--------|----------|---------------------------------|
| GET    | `/`      | Health check → `{"status":"ok"}`|
| POST   | `/reset` | Reset env, returns Observation  |
| POST   | `/step`  | Advance one step with Action    |
| GET    | `/state` | Get raw current state dict      |

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
uvicorn app:app --host 0.0.0.0 --port 7860
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
[START]
[STEP] step=1 decision=decrease_thrust reward=0.3000 reason=High altitude, slowing descent
[STEP] step=2 decision=stabilize reward=0.3000 reason=High wind detected
...
[END] total_reward=1.42
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

## Resource Requirements

- vCPU: 2
- Memory: 8 GB
- Inference runtime: < 20 minutes

---

## Project Structure

```
rocket-landing-openenv/
├── app.py            # FastAPI server (OpenEnv HTTP API)
├── environment.py    # Core environment (reset/step/state + models)
├── tasks.py          # Task graders (easy / medium / hard)
├── inference.py      # Baseline inference script
├── openenv.yaml      # OpenEnv specification manifest
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Team

- Jainam Dedhia (Team Lead)
- Maher Dhami
- Tirth Shah