"""
app.py — FastAPI server for the Log Analysis RL Environment.

Required endpoints:
  GET  /health    — health check
  GET  /tasks     — task list + action schema
  POST /reset     — start episode
  POST /step      — submit action, get reward
  GET  /state     — internal env state
  GET  /grader    — final episode score (after done=true)
  POST /baseline  — run GPT baseline agent, return scores

Run locally:
  uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import threading
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from models import LogObservation, AnalysisAction, StepReward, EnvState
from server.logScan_env_environment import LogAnalysisEnvironment

app = FastAPI(
    title="Log Analysis + Incident Root Cause Agent",
    version="1.0.0",
)

#one environment instance per task level
envs = {
    "easy":   LogAnalysisEnvironment(task_id="easy"),
    "medium": LogAnalysisEnvironment(task_id="medium"),
    "hard":   LogAnalysisEnvironment(task_id="hard"),
}

#cache last grader result per task (cleared on reset)
last_grader = {"easy": None, "medium": None, "hard": None}

#baseline runs once and caches the result
baseline_cache = None
baseline_lock = threading.Lock()


def get_env(task_id):
    if task_id not in envs:
        raise HTTPException(400, f"Unknown task_id '{task_id}'. Use: easy, medium, hard")
    return envs[task_id]


# /health

@app.get("/health")
def health():
    return {"status": "healthy", "version": "1.0.0", "tasks": list(envs.keys())}



# /tasks - task list + action schema

@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {
                "task_id": "easy",
                "description": "Detect anomaly and classify error type. Navigate logs, then call classify_error.",
                "difficulty": "easy",
                "max_steps": 15,
                "max_reward": 1.0,
                "reward_breakdown": {"anomaly": 0.2, "classification": 0.3},
            },
            {
                "task_id": "medium",
                "description": "Classify error + identify root cause line. Use filter/search/mark_root_cause.",
                "difficulty": "medium",
                "max_steps": 20,
                "max_reward": 1.0,
                "reward_breakdown": {"anomaly": 0.2, "classification": 0.3, "root_cause": 0.3},
            },
            {
                "task_id": "hard",
                "description": "Full pipeline: classify, root cause, and suggest fix.",
                "difficulty": "hard",
                "max_steps": 30,
                "max_reward": 1.0,
                "reward_breakdown": {"anomaly": 0.2, "classification": 0.3, "root_cause": 0.3, "fix": 0.2},
            },
        ],
        "action_schema": {
            "action_type": {
                "type": "string",
                "required": True,
                "options": [
                    "next_log",         # move window forward
                    "prev_log",          # move window backward
                    "jump_to_index",   # jump to line (needs target_index: int)
                    "filter_by_level",      # jump to first ERROR/WARN/INFO (needs filter_level: str)
                    "search_logs",      # search keyword, jump to first hit (needs search_keyword: str)
                    "classify_error",   # declare error type (needs error_type: str)
                    "mark_root_cause",   # identify root cause line (needs root_cause_line_id + explanation)
                    "suggest_fix",      # propose fix (needs fix_suggestion: str)
                ],
            },
            "target_index":          {"type": "int",    "for": "jump_to_index"},
            "filter_level":          {"type": "string", "for": "filter_by_level", "options": ["INFO", "WARNING", "ERROR"]},
            "search_keyword":        {"type": "string", "for": "search_logs"},
            "error_type":            {"type": "string", "for": "classify_error", "options": ["timeout", "memory", "auth", "network", "none"]},
            "root_cause_line_id":    {"type": "int",    "for": "mark_root_cause"},
            "root_cause_explanation":{"type": "string", "for": "mark_root_cause"},
            "fix_suggestion":        {"type": "string", "for": "suggest_fix"},
        },
        "example_actions": {
            "filter_errors":  {"action_type": "filter_by_level", "filter_level": "ERROR"},
            "search":         {"action_type": "search_logs", "search_keyword": "timeout"},
            "classify":       {"action_type": "classify_error", "error_type": "memory"},
            "mark_cause":     {"action_type": "mark_root_cause", "root_cause_line_id": 8, "root_cause_explanation": "OOM caused by unbounded cache after Redis failure"},
            "fix":            {"action_type": "suggest_fix", "fix_suggestion": "Restore Redis and add LRU eviction with 512MB cap"},
        },
    }



# /reset

class ResetRequest(BaseModel):
    task_id: str = "easy"


@app.post("/reset", response_model=LogObservation)
def reset(req: ResetRequest):
    env = get_env(req.task_id)
    obs = env.reset()
    last_grader[req.task_id] = None
    return obs


# /step

class StepRequest(BaseModel):
    task_id: str = "easy"
    action: AnalysisAction


class StepResponse(BaseModel):
    observation: LogObservation
    reward: StepReward
    done: bool
    info: dict


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    env = get_env(req.task_id)
    try:
        obs, reward, done, info = env.step(req.action)
    except RuntimeError as e:
        raise HTTPException(400, str(e))

    if done and env.last_result:
        last_grader[req.task_id] = env.last_result.model_dump()

    return StepResponse(observation=obs, reward=reward, done=done, info=info)



# /state

@app.get("/state", response_model=EnvState)
def state(task_id: str = Query(default="easy")):
    return get_env(task_id).state()



# /grader - returns final episode score 

@app.get("/grader")
def grader(task_id: str = Query(default="easy")):
    
    if task_id not in last_grader:
        raise HTTPException(400, f"Unknown task_id '{task_id}'")
    result = last_grader[task_id]

    if result is None:
        raise HTTPException(404, f"No completed episode for '{task_id}'. Call /reset then /step until done=true.")
    return result



# /baseline - trigger GPT baseline agent

@app.post("/baseline")
def baseline():
    global baseline_cache

    if baseline_cache is not None:

        return {**baseline_cache, "cached": True}

    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(503, "OPENAI_API_KEY not set. Add it as a Space secret.")

    from baseline import run_baseline
    try:
        with baseline_lock:

            if baseline_cache is not None:
                return {**baseline_cache, "cached": True}
            result = run_baseline(base_url="http://localhost:8000", episodes_per_task=3)
            baseline_cache = result

        return {**result, "cached": False}
    except Exception as e:
        raise HTTPException(500, f"Baseline failed: {e}")
