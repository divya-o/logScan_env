from pydantic import BaseModel
from typing import Optional, List, Literal


class LogObservation(BaseModel):
    task_id: str
    log_lines: List[str]            # current visible window of logs
    window_start: int               # index of first line shown
    total_lines: int               # total lines in this scenario
    instruction: str             # what the agent must do
    steps_taken: int
    max_steps: int
    done: bool = False


class AnalysisAction(BaseModel):
    action_type: Literal[
        "next_log",
        "prev_log",
        "jump_to_index",
        "filter_by_level",
        "search_logs",
        "classify_error",
        "mark_root_cause",
        "suggest_fix",
    ]
    # used by jump_to_index
    target_index: Optional[int] = None
    # used by filter_by_level
    filter_level: Optional[str] = None  
    # used by search_logs
    search_keyword: Optional[str] = None
    # used by classify_error
    error_type: Optional[str] = None     
    # used by mark_root_cause
    root_cause_line_id: Optional[int] = None
    root_cause_explanation: Optional[str] = None
    # used by suggest_fix
    fix_suggestion: Optional[str] = None


class StepReward(BaseModel):
    total: float                           # step reward, can be negative
    anomaly_score: float = 0.0            # +0.2 first correct classify
    classification_score: float = 0.0    # +0.3 correct error type
    root_cause_score: float = 0.0         # +0.0 to +0.3 keyword match
    fix_score: float = 0.0               # +0.0 to +0.2 keyword match
    step_penalty: float = -0.05           # every step costs this
    penalty: float = 0.0                  # extra penalty 
    feedback: str = ""
    done: bool = False


class EpisodeResult(BaseModel):
    task_id: str
    scenario_id: str
    total_score: float    # final score 0.0 to 1.0
    steps_taken: int
    feedback: str


class EnvState(BaseModel):
    task_id: str
    scenario_id: str
    steps_taken: int
    max_steps: int
    done: bool
    cumulative_reward: float
    anomaly_done: bool
    classification_done: bool
    root_cause_done: bool
    fix_done: bool
