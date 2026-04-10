---
title: LogScan-Env
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
tags:
  - openenv
---

# LogScan-Env 
 
## Multi-Step Interactive Debugging Intelligence Environment

OpenEnv Reinforcement Learning(RL) complaint debugging reasoning & inteliigence environment where an ML & AI  agent learns and debugs real system incidents by navigating noisy logs, applying filters and searches, classifying errors, identifying root causes, and suggesting fixes. Simulated to developer's debugging workflow. 

  [ TRAIN YOUR AGENT TO DEBUG ]


## Motivation 

Production incident response is one of the most demanding sequential decision-making
tasks in software operations. From an on-call engineer, developer facing error windows to a SRE debugging the issue must:

- Navigate hundreds of noisy log lines with no global view
- Apply targeted search and filter strategies to isolate signals from noise
- Form and revise hypotheses under step-cost pressure
- Identify the single causal log line from a sea of correlated symptoms
- Produce an actionable fix grounded in the specific failure mode


This environment  rewards investigation strategy through a
dense per-step signal and penalizes inefficiency via a step cost, creating a genuine
exploration-exploitation tradeoff. It is designed for training and evaluating agents
on the full incident-response pipeline: **navigate → filter → classify → diagnose →
remediate** — with each stage independently measurable.


- Partial observability.
 
    Only a 6-line window of a 20–25 line log is visible at any step; the agent must reconstruct the full incident from fragments.
- Dense but shaped reward.
  
    Every step returns a signal - −0.05 per step, −0.10 for irrelevant actions, −0.30 for a wrong root cause - so the agent must learn which actions are worth taking and when.
  
- Three difficulty tiers.
  
    The same log corpus across Easy, Medium, and Hard, with increasing decision depth: classify → identify root cause - suggest fix.



## Environment Design

### Episode Flow

```
POST /reset    →  LogObservation   (first 6-line window; episode begins)
POST /step     →  LogObservation   (updated window)
               +  StepReward       (per-step dense reward signal)
               +  done: bool
               +  info: dict
               ...  (repeat up to max_steps)
GET  /grader   →  EpisodeResult    (final score 0.0–1.0; call after done=true)
```

Each episode is a single incident scenario. The agent does not have access to the full
log - it sees a **sliding window of 6 lines** and must navigate deliberately. The
episode terminates when the agent calls all required decision actions for its task tier,
or when `max_steps` is exhausted.



## Observation Space

Each step returns a `LogObservation` with the following fields:

| Field                 | Type             | Description                                                    |
|-----------------------|------------------|----------------------------------------------------------------|
| `task_id`             | `string`         | `"easy"`, `"medium"`, or `"hard"`                              |
| `current_window`      | `list[string]`   | The 6 log lines currently visible, prefixed with `[NNN]`       |
| `window_start`        | `integer`        | Index of the first line in the current window                  |
| `total_log_lines`     | `integer`        | Total lines in this incident scenario                          |
| `active_filter`       | `string \| null` | Currently active severity filter (`INFO`, `WARNING`, `ERROR`)  |
| `last_search_results` | `list[int]`      | Line indices matching the most recent `search_logs` call       |
| `task_description`    | `string`         | Plain-English objective for this episode                       |
| `steps_taken`         | `integer`        | Steps used so far in this episode                              |
| `max_steps`           | `integer`        | Maximum steps before forced termination                        |
| `done`                | `boolean`        | `true` when the episode is complete                            |
| `detected_signals`    | `list[string]`   | Error keywords accumulated from all visited log lines so far   |

The `detected_signals` field is cumulative — it grows as the agent navigates — and is
intended to support agents that maintain a running hypothesis across steps without
re-reading prior windows.



## Action Space

The agent submits one action per step via `POST /step`. The `action_type` field is
always required; additional fields depend on the action type.

### Navigation Actions

| Action Type      | Extra Fields        | Description                              |
|------------------|---------------------|------------------------------------------|
| `next_log`       | —                   | Advance the window forward by 6 lines    |
| `prev_log`       | —                   | Move the window backward by 6 lines      |
| `jump_to_index`  | `target_index: int` | Jump the window to a specific line index |

Navigation at a boundary (e.g., `next_log` at end of log) incurs a −0.10 irrelevant
action penalty on top of the standard step cost.

### Filter & Search Actions

| Action Type       | Extra Fields                | Description                                        |
|-------------------|-----------------------------|----------------------------------------------------|
| `filter_by_level` | `filter_level: str \| null` | Filter view to `INFO`, `WARNING`, `ERROR`, or clear|
| `search_logs`     | `search_keyword: str`       | Search all lines; jump the window to the first hit |

`filter_by_level` restricts which lines appear in the window without altering the
underlying log. `search_logs` scans the full unfiltered log and populates
`last_search_results` with all matching indices; the window jumps to the first match.
An empty search (no keyword match) incurs a −0.10 penalty.

### Decision Actions (scored)

| Action Type       | Extra Fields                                              | Description                         |
|-------------------|-----------------------------------------------------------|-------------------------------------|
| `classify_error`  | `error_type: timeout\|memory\|auth\|network\|none`        | Declare the incident's error type   |
| `mark_root_cause` | `root_cause_line_id: int`, `root_cause_explanation: str`  | Identify the root-cause log line    |
| `suggest_fix`     | `fix_suggestion: str`                                     | Propose an actionable remediation   |

Decision actions are the only actions that generate positive reward. They are also
irreversible within an episode — the agent cannot re-classify or re-mark after
submission. A wrong `mark_root_cause` (incorrect line index) immediately incurs −0.30
with no recovery path for that component, creating a strong incentive for the agent
to reach high confidence before committing.



## Tasks

### Task 1 — Easy: Anomaly Detection (max 15 steps)

* The agent must detect whether an anomaly is present and classify its error type. The episode ends the moment `classify_error` is called.

* **Required actions:** `classify_error`

 **Expected strategy:** `filter_by_level(ERROR)` → scan window → `classify_error`

* **Score ceiling:** 0.5 &nbsp;(anomaly detection 0.2 + classification 0.3)



### Task 2 — Medium: Root Cause Identification (max 20 steps)

* Beyond classification, the agent must pinpoint the specific log line that caused the incident. Both `classify_error` and `mark_root_cause` must be called.

* **Required actions:** `classify_error` + `mark_root_cause`

* **Expected strategy:** `filter_by_level(ERROR)` → `search_logs(keyword)` → identify
causal line → `classify_error` → `mark_root_cause(line_id, explanation)`

* **Score ceiling:** 0.8 &nbsp;(anomaly 0.2 + classification 0.3 + root cause up to 0.3)



### Task 3 — Hard: Full Incident Pipeline (max 30 steps)

* The complete workflow. The agent must classify, diagnose, and propose a concrete fix.
* The step cost makes brute-force navigation increasingly costly — agents that reach the root cause in fewer steps receive an efficiency bonus.

* **Required actions:** `classify_error` + `mark_root_cause` + `suggest_fix`

* **Expected strategy:** filter → search → classify → mark root cause → suggest fix.
Target under 15 steps to qualify for the efficiency bonus.

* **Score ceiling:** 1.0 &nbsp;(anomaly 0.2 + classification 0.3 + root cause 0.3 + fix 0.2)



## Reward Function

### Design Philosophy

The reward function is built around three principles:

- *Dense reward signal.* Every step returns a reward, giving gradient-based methods a continuous learning signal rather than a sparse terminal one.
- *Efficiency penalized.* Step costs and irrelevant-action penalties force the policy to learn when to act, not just what to do.
- *Partial credit*. Root cause and fix scores use keyword overlap, so a correct diagnosis with imprecise wording still yields a meaningful gradient signal.

### Per-Step Reward Table

| Signal                    | Value           | Trigger Condition                                                                        |
|---------------------------|-----------------|------------------------------------------------------------------------------------------|
| Step cost                 | −0.05           | Every step, without exception                                                            |
| Irrelevant action         | −0.10           | Navigation at boundary; empty search result; redundant filter                            |
| Wrong root cause          | −0.30           | `mark_root_cause` with an incorrect `root_cause_line_id`                                 |
| Anomaly detection         | +0.20           | First `classify_error` call on an incident that contains an anomaly                     |
| Error classification      | +0.30           | `classify_error` with the correct `error_type`                                           |
| Root cause identification | +0.15 to +0.30  | `mark_root_cause` scored by keyword overlap between explanation and ground-truth line    |
| Fix suggestion            | +0.10 to +0.20  | `suggest_fix` scored by keyword overlap between suggestion and ground-truth remediation  |

### Episode Score Calculation

The grader accumulates the *best* value achieved for each component across all steps
in the episode, then sums them:

```
episode_score = anomaly_score          # 0.0 or 0.20
              + classification_score   # 0.0 or 0.30
              + root_cause_score       # 0.0 to 0.30  (keyword overlap)
              + fix_score              # 0.0 to 0.20  (keyword overlap)
              + efficiency_bonus       # > 0.0 if solved in < 50% of max_steps
```

The efficiency bonus is only awarded if all required component scores are non-zero —
i.e., the agent genuinely solved the task, not merely exhausted its step budget.

### Example: Full Hard Episode via curl

```bash
# 1. Start episode
curl -s -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "hard"}' | python -m json.tool

# 2. Filter to error-level logs
curl -s -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"task_id": "hard", "action": {"action_type": "filter_by_level", "filter_level": "ERROR"}}'

# 3. Search for the failure keyword
curl -s -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"task_id": "hard", "action": {"action_type": "search_logs", "search_keyword": "timeout"}}'

# 4. Classify the error type
curl -s -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"task_id": "hard", "action": {"action_type": "classify_error", "error_type": "timeout"}}'

# 5. Mark the root cause line
curl -s -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "hard",
    "action": {
      "action_type": "mark_root_cause",
      "root_cause_line_id": 3,
      "root_cause_explanation": "Sequential scan on events table with no index on date column causes full table scan of 14M rows, exhausting the DB connection pool."
    }
  }'

# 6. Propose a fix
curl -s -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "hard",
    "action": {
      "action_type": "suggest_fix",
      "fix_suggestion": "Create index on events(date). Run EXPLAIN ANALYZE to verify the query plan. Schedule VACUUM ANALYZE to update table statistics."
    }
  }'

# 7. Retrieve final episode score
curl -s "http://localhost:8000/grader?task_id=hard"
```



## Setup & Deployment

### Local Development

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/logScan-env
cd logScan-env
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

### Docker

```bash
docker build -t logScan-env .
docker run -p 8000:8000 logScan-env
```

### Validate
```
openenv validate .
openenv validate --url http://localhost:8000
```

### Run the Baseline Inference Script

```bash
export API_BASE_URL=https://api.openai.com/v1
export OPENAI_API_KEY="sk-..."
export ENV_BASE_URL="http://localhost:8000"
python inference.py
OR
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="your-model-name"
export HF_TOKEN="your-hf-token"
export ENV_BASE_URL="http://localhost:8000"
# Output written to: inference_scores.json
```

### Trigger Inference via API

```bash
# Requires OPENAI_API_KEY set as a Space secret
curl -X POST http://localhost:8000/inference
```

### Deploy to HF Spaces

```bash
# 1. Update the author field in openenv.yaml to your HF username
# 2. Authenticate
huggingface-cli login
# 3. Push
openenv push --repo-id YOUR_USERNAME/logScan-env
# 4. Add secret: Space → Settings → Variables and Secrets → OPENAI_API_KEY(if using OpenAI API )
```



## Inference Scores & Analysis

The inference agent uses `gpt-4o-mini` at `temperature=0.0` with a fixed prompt
strategy: filter to `ERROR`, search for the error keyword, classify, mark root cause,
suggest fix. Evaluated over 3 episodes per task.

| Task        | Strategy                                                | Mean Score |
|-------------|---------------------------------------------------------|------------|
| Easy        | Filter → classify                                       | ~0.50      |
| Medium      | Filter → search → classify → root cause                 | ~0.55      |
| Hard        | Full pipeline (filter → search → classify → mark → fix) | ~0.45      |
| **Overall** |                                                         | **~0.50**  |



## Submission validations

-  `author` in `openenv.yaml` updated to your HF username
-  `GET /health` returns `{"status": "healthy"}`
-  `POST /reset` returns `LogObservation` with `current_window`
-  `POST /step` returns `StepReward` with `total` in [−1.0, 1.0]
-  `GET /grader` returns `EpisodeResult` after episode completes
-  `GET /tasks` includes `action_schema` with all  action types
-  `POST /baseline` triggers inference and returns structured scores
-  `openenv.yaml` passes `openenv validate .`
-  `docker build` completes without errors
-  `python inference.py` completes and writes `inference_scores.json`
-  `OPENAI_API_KEY` set as HF Space secret


## License

MIT

