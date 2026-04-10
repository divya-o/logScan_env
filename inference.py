#usage:#export API_BASE_URL="https://router.huggingface.co/v1"
  #export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
  #export HF_TOKEN="hf_..."
  #export ENV_BASE_URL="http://localhost:8000"   # optional, defaults to localhost
  #python inference.py

import os
import sys
import json
import requests
from openai import OpenAI



# cconfig - read from required competition environment variables

def get_config():
    api_base_url=os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1" )
    model_name= os.environ.get ("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct") 
    hf_token= os.environ.get("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")


    missing= [k for k, v in[
        ("API_BASE_URL",api_base_url),
        ("MODEL_NAME", model_name),
        ("HF_TOKEN", hf_token ),
    ] if not v]

    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Set them before running:\n"
            "  export API_BASE_URL=...\n"
            "  export MODEL_NAME=...\n"
            "  export HF_TOKEN=..."
        )

    env_base_url=os.environ.get("ENV_BASE_URL", "http://localhost:8000")
    return api_base_url, model_name, hf_token, env_base_url


def get_client(api_base_url, hf_token) :
    return OpenAI(api_key=hf_token, base_url=api_base_url )



# System prompt

SYSTEM_PROMPT="""You are an expert Site Reliability Engineer (SRE).

You interact with a log analysis environment step by step.
Each turn you see the current log window and must pick ONE action.

AVAILABLE ACTIONS (respond with ONLY the JSON, no markdown):
  {"action_type":"next_log"}
  {"action_type": "prev_log" }
  {"action_type": "jump_to_index", "target_index": <int>}
  {"action_type": "filter_by_level", "filter_level": "ERROR"}
  {"action_type": "search_logs", "search_keyword": "<keyword>"}
  {"action_type": "classify_error", "error_type": "<timeout|memory|auth|network|none>"}
  {"action_type": "mark_root_cause", "root_cause_line_id": <int>, "root_cause_explanation": "<why>"}
  {"action_type": "suggest_fix", "fix_suggestion": "<concrete steps>"}

STRATEGY (follow this order for efficiency):
1. filter_by_level ERROR  - jump to errors immediately
2. search_logs <keyword>  - find relevant signals (timeout, OOM, refused, crash...)
3. classify_error         - declare the error type
4. mark_root_cause        - identify the exact [NNN] line that caused it
5. suggest_fix            - propose concrete resolution steps

Each step costs -0.05 reward. Be efficient. Respond ONLY with valid JSON.
"""



# Deterministic fallback strategy
 

FALLBACK_SEQUENCE=[
    {"action_type": "filter_by_level", "filter_level":"ERROR" },
    {"action_type": "search_logs","search_keyword": "error"},
    {"action_type": "next_log" },
    {"action_type": "classify_error","error_type": "timeout"},
    {"action_type": "mark_root_cause", "root_cause_line_id":0,
     "root_cause_explanation": "First error line identified as root cause"},
    {"action_type": "suggest_fix",
     "fix_suggestion": "Review error logs and apply appropriate fix based on error type"},
]


def get_fallback_action(step_num):                          #based on step number 
    idx=min(step_num, len(FALLBACK_SEQUENCE) - 1)
    return FALLBACK_SEQUENCE[idx]



# Parse LLM response

def parse_action(raw) :                      #parse json from llm response
    raw=raw.strip()
    if raw.startswith("```"):
        parts=raw.split("```")
        raw=parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw=raw[4: ]
    return json.loads( raw.strip() )



# Structured log helpers - required stdout format
def log_start(task_id, episode, model_name, env_base_url) :
    print(f"[START] task_id={task_id} episode={episode} model={model_name} env={env_base_url}")
    sys.stdout.flush()


def log_step(step_num, action_type, reward, done, error=None) :          #print [STEP] block after every env step, req. fields: step, action, reward, done, error
    error_str=str(error) if error else "none"
    print(
        f"[STEP] step ={step_num} "
        f"action= {action_type } "
        f"reward= {reward:.4f} "
        f"done= { str(done).lower()} "
        f"error= {error_str}"
    )
    sys.stdout.flush()


def log_end(task_id, episode, final_score, steps_taken, success ):               #print [END] block after each episode completes, req. fields: task_id, episode, score, steps, success
    print(
        f"[END] task_id= {task_id} "
        f"episode ={episode} "
        f"score= {final_score:.4f} "
        f"steps= {steps_taken} "
        f"success= {str(success).lower()}"
    )
    sys.stdout.flush()




def run_episode (task_id, episode_num, client, model_name,env_base_url ) :                #runs one full multi-step episode
    log_start(task_id, episode_num, model_name, env_base_url)

    #reset env --
    try:
        r=requests.post(
            f"{env_base_url}/reset",
            json={"task_id": task_id},
            timeout=15,
        )
        r.raise_for_status()
        obs=r.json()
    except Exception as e:
        # Cannot even reset — log and return zero score
        log_step(0, "reset_failed", 0.0, True, error=str(e))
        log_end(task_id, episode_num, 0.01, 0, success=False)
        return {"score": 0.01, "steps": 0, "success": False}

    history=[]
    step_num=0
    done=False
    cumulative_reward=0.0
    max_steps=obs.get("max_steps", 30)

    while not done and step_num < max_steps:
        step_num += 1
        llm_error=None
        action_type="unknown"

        #  Build user message from current observation 
        window="\n".join(obs.get("log_lines", []))
        user_msg=(
            f"TASK: {obs.get('instruction', '')}\n\n"
            f"LOG WINDOW (lines {obs.get('window_start', 0)}-"
            f"{obs.get('window_start', 0) + 6}, "
            f"total={obs.get('total_lines', 0)}):\n"
            f"{window}\n\n"
            f"Step {obs.get('steps_taken', 0)}/{max_steps} | "
            f"Cumulative reward so far: {cumulative_reward:.3f}\n"
            "Choose your next action (JSON only):"
        )
        history.append({"role": "user", "content": user_msg})

        #  Ask LLM for next action 
        try:
            resp=client.chat.completions.create(
                model=model_name,
                temperature=0.0,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history[-6:],
            )
            raw=resp.choices[0].message.content
            history.append({"role": "assistant", "content": raw})
            action=parse_action(raw)
            action_type=action.get("action_type", "unknown")

        except Exception as e:
            # LLM failed or returned unparseable output — use deterministic fallback
            llm_error=f"llm_error:{type(e).__name__}"
            action=get_fallback_action(step_num - 1)
            action_type=action["action_type"] + "(fallback)"
            history.append({"role": "assistant", "content": json.dumps(action)})

        #  Submit action to environment 
        step_reward=0.0
        step_error=llm_error

        try:
            step_resp=requests.post(
                f"{env_base_url}/step",
                json={"task_id": task_id, "action": action},
                timeout=15,
            )
            step_resp.raise_for_status()
            data=step_resp.json()

            # Read reward properly from step response
            reward_data=data.get("reward", {})
            step_reward=reward_data.get("total", 0.0)
            feedback   =reward_data.get("feedback", "")

            cumulative_reward += step_reward
            obs =data["observation"]
            done=data["done"]

            # Append feedback to history so agent can learn from it
            history.append({
                "role": "user",
                "content": f"[ENV FEEDBACK] reward={step_reward:.4f} | {feedback}"
            })

        except Exception as e:
            # Environment call failed
            step_error=f"env_error:{type(e).__name__}"
            done=True   # treat env failure as episode end to avoid stalling

        #  Log this step 
        log_step(step_num, action_type, step_reward, done, error=step_error)

    #  Get final episode score from /grader 
    final_score=0.01
    try:
        g=requests.get(
            f"{env_base_url}/grader",
            params={"task_id": task_id},
            timeout=10,
        )
        if g.status_code == 200:
            final_score=max(0.01, min(0.99, raw))
    except Exception:
        # If grader call fails, fall back to estimating from cumulative reward
        final_score=max(0.0, min(0.99, cumulative_reward))

    success=final_score > 0.01
    log_end(task_id, episode_num, final_score, step_num, success)

    return {"score": final_score, "steps": step_num, "success": success}



# run_inference() — importable by server/app.py for POST /inference

def run_inference(base_url=None, episodes_per_task=3):
    """
    Run inference agent on all 3 tasks.
    Called by server/app.py /inference endpoint AND by main() below.

    Returns structured dict for evaluator:
    {
      "model": "...",
      "scores": {"easy": 0.5, "medium": 0.55, "hard": 0.45},
      "overall": 0.5,
      "details": { ... }
    }
    """
    api_base_url, model_name, hf_token, env_base_url=get_config()
    if base_url:
        env_base_url=base_url

    client=get_client(api_base_url, hf_token)

    scores ={}
    details={}

    for task_id in ("easy", "medium", "hard"):
        ep_scores  =[]
        ep_details =[]

        for ep in range(1, episodes_per_task + 1):
            result=run_episode(task_id, ep, client, model_name, env_base_url)
            ep_scores.append(result["score"])
            ep_details.append(result)

        mean=round(sum(ep_scores) / len(ep_scores), 4) if ep_scores else 0.01
        scores[task_id] =mean
        details[task_id]={
            "mean_score":     mean,
            "episodes_run":   len(ep_scores),
            "episode_scores": ep_scores,
            "episodes":       ep_details,
        }

    overall=round(sum(scores.values()) / len(scores), 4)

    return {
        "model":   model_name,
        "scores":  scores,
        "overall": overall,
        "details": details,
    }



# CLI entrypoint

if __name__ == "__main__":
    # Validate env vars first — fail clearly if missing
    try:
        api_base_url, model_name, hf_token, env_base_url=get_config()
    except EnvironmentError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] model={model_name} env={env_base_url}")

    # Health check
    try:
        h=requests.get(f"{env_base_url}/health", timeout=5)
        h.raise_for_status()
        print(f"[INFO] health={h.json().get('status', 'ok')}")
    except Exception as e:
        print(f"[ERROR] Environment not reachable: {e}", file=sys.stderr)
        sys.exit(1)

    # Run inference
    result=run_inference(base_url=env_base_url, episodes_per_task=3)

    # Print final scores in standardized format
    print("\n[SCORES]")
    for task_id, score in result["scores"].items():
        bar="#" * int(score * 20)
        print(f"  task={task_id} score={score:.4f} [{bar:<20}]")
    print(f"  overall={result['overall']:.4f}")

    # Save scores file for evaluator
    output={
        "model":   result["model"],
        "scores":  result["scores"],
        "overall": result["overall"],
    }
    with open("inference_scores.json", "w") as f:
        json.dump(output, f, indent=2)
    print("[INFO] saved inference_scores.json")