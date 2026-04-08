#grader.py - Scores agent actions and final episodes.
"""
functions:
  grade_step()    - called every step, returns StepReward (dense signal)
  grade_episode()  called at episode end
 

Reward weights (match problem statement):
  +0.2  anomaly detection
  +0.3  error classification
  +0.3  root cause 
  +0.2  fix suggestion 
  -0.05 every step 
  -0.1  irrelevant action 
  -0.3  wrong root cause line
"""

from models import AnalysisAction, StepReward, EpisodeResult


def keyword_score(text, keywords, max_score):
    """Score based on how many keywords appear in text. 0/1/2/3+ hits = 0/50/75/100%."""
    if not text or not isinstance(text, str):
        return 0.0
    text = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text)
    if hits == 0:
        return 0.0
    if hits == 1:
        return round(max_score * 0.5, 4)
    if hits == 2:
        return round(max_score * 0.75, 4)
    return max_score


def grade_step(action, ground_truth, state):     #score one agent action.

    a_score = 0.0
    c_score = 0.0
    rc_score = 0.0
    fix_score = 0.0
    step_pen = -0.05
    extra_pen = 0.0
    feedback = f"step_penalty=-0.05"
    done = False

    atype = action.action_type

    #nvigation 
    if atype in ("next_log", "prev_log", "jump_to_index"):
        at_end = state["window_start"] + 6 >= state["total_lines"]
        at_start = state["window_start"] == 0
        if (atype == "next_log" and at_end) or (atype == "prev_log" and at_start):
            extra_pen = -0.1
            feedback += " | irrelevant navigation at boundary (-0.1)"
        else:
            feedback += f" | {atype}: ok"

    #filter 
    elif atype == "filter_by_level":
        feedback += f" | filter_by_level: {action.filter_level}"

    #search 
    elif atype == "search_logs":
        if not action.search_keyword:
            extra_pen = -0.1
            feedback += " | empty search keyword (-0.1)"
        else:
            feedback += f" | search: '{action.search_keyword}'"

    #classify error (anomaly + type) 
    elif atype == "classify_error":
        if not state["anomaly_done"]:
            correct = ground_truth["anomaly_detected"]
            agent_says_anomaly = action.error_type != "none"

            if agent_says_anomaly == correct:
                a_score = 0.2
                feedback += " | anomaly: correct (+0.2)"
            else:
                feedback += " | anomaly: wrong (+0.0)"

        if not state["classification_done"]:
            correct_type = ground_truth.get("error_type", "none")
            agent_type = action.error_type or "none"
            if agent_type == correct_type:
                c_score = 0.3
                feedback += f" | error_type: correct '{correct_type}' (+0.3)"
            else:
                feedback += f" | error_type: wrong got '{agent_type}' expected '{correct_type}'"

    #mark root cause 
    elif atype == "mark_root_cause":
        if not state["root_cause_done"]:
            correct_lines = set(ground_truth.get("root_cause_line_ids", []))
            agent_line = action.root_cause_line_id
            explanation = action.root_cause_explanation or ""
            rc_keywords = ground_truth.get("root_cause_keywords", [])

            line_ok = (agent_line in correct_lines) if correct_lines else True
            expl_score = keyword_score(explanation, rc_keywords, 0.3)

            if line_ok:
                rc_score = expl_score if expl_score > 0 else 0.3  # line alone = full
                feedback += f" | root_cause line {agent_line}: correct (+{rc_score:.3f})"
            else:
                extra_pen = -0.3
                feedback += f" | root_cause line {agent_line}: WRONG (-0.3), expected {correct_lines}"
        else:
            feedback += " | root_cause already marked"

    #suggest fix 
    elif atype == "suggest_fix":
        if not state["fix_done"]:
            fix_kws = ground_truth.get("fix_keywords", [])
            fix_text = action.fix_suggestion or ""
            fix_score = keyword_score(fix_text, fix_kws, 0.2)
            feedback += f" | fix suggestion: +{fix_score:.3f}"
        else:
            feedback += " | fix already suggested"

    #check if episode should end
    task = state["task_level"]
    now_anomaly = state["anomaly_done"] or a_score > 0
    now_class = state["classification_done"] or c_score > 0
    now_rc = state["root_cause_done"] or rc_score > 0
    now_fix = state["fix_done"] or fix_score > 0

    if task == "easy" and now_anomaly:
        done = True
    elif task == "medium" and now_anomaly and now_class and now_rc:
        done = True
    elif task == "hard" and now_anomaly and now_class and now_rc and now_fix:
        done = True

    if state["steps_taken"] + 1 >= state["max_steps"]:
        done = True
        feedback += " | max_steps reached"

    total = round(a_score + c_score + rc_score + fix_score + step_pen + extra_pen, 4)
    total = max(-1.0, min(1.0, total))

    return StepReward(
        total=total,
        anomaly_score=a_score,
        classification_score=c_score,
        root_cause_score=rc_score,
        fix_score=fix_score,
        step_penalty=step_pen,
        penalty=extra_pen,
        feedback=feedback,
        done=done,
    )


def grade_episode(scenario_id, task_id, best, steps, max_steps):
    a = min(0.2, best.get("anomaly", 0.0))
    c = min(0.3, best.get("classification", 0.0))
    r = min(0.3, best.get("root_cause", 0.0))
    f = min(0.2, best.get("fix", 0.0))
    raw = a + c + r + f

    #small bonus for solving efficiently
    bonus = 0.0
    if raw >= 0.8 and steps <= max_steps // 2:
        bonus = min(0.1, 1.0 - raw)

    total = round(min(1.0, raw + bonus), 4)

    return EpisodeResult(
        task_id=task_id,
        scenario_id=scenario_id,
        total_score=total,
        steps_taken=steps,
        feedback=(
            f"anomaly={a:.2f}/0.2 | classification={c:.2f}/0.3 | "
            f"root_cause={r:.2f}/0.3 | fix={f:.2f}/0.2 | "
            f"steps={steps}/{max_steps}"
            + (f" | efficiency_bonus=+{bonus:.2f}" if bonus > 0 else "")
        ),
    )
