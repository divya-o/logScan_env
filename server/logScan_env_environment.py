from models import LogObservation, AnalysisAction, StepReward, EnvState, EpisodeResult
from server.tasks import TASKS, WINDOW_SIZE, MAX_STEPS
from server.grader import grade_step, grade_episode


class LogAnalysisEnvironment:
    def __init__(self, task_id="easy", seed=42):
        assert task_id in ("easy", "medium", "hard")
        self.task_id = task_id
        self._scenarios = TASKS[task_id]
        self._max_steps = MAX_STEPS[task_id]

        # Episode state
        self._scenario = {}
        self._log_lines = []
        self._window_start = 0
        self._steps = 0
        self._done = True
        self._total_episodes = 0
        self._scenario_index = 0

        # What has the agent completed this episode
        self._anomaly_done = False
        self._class_done = False
        self._rc_done = False
        self._fix_done = False

      
        self._best = {"anomaly": 0.0, "classification": 0.0, "root_cause": 0.0, "fix": 0.0}

        self._cumulative = 0.0

        #last episode result
        self.last_result = None

    def reset(self):
        self._scenario_index = self._total_episodes % len(self._scenarios)
        self._scenario = self._scenarios[self._scenario_index]
        self._log_lines = self._scenario["log_lines"]
        self._window_start = 0
        self._steps = 0
        self._done = False
        self._anomaly_done = False

        self._class_done = False
        self._rc_done = False
        self._fix_done = False
        self._best = {"anomaly": 0.0, "classification": 0.0, "root_cause": 0.0, "fix": 0.0}
        self._cumulative = 0.0
        self._total_episodes += 1
        self.last_result = None
        return self._make_obs()

    def step(self, action: AnalysisAction):
        if self._done:
            raise RuntimeError("Call reset() before step().")

        self._apply(action)

        state_snapshot = {
            "task_level": self.task_id,
            "anomaly_done": self._anomaly_done,
            "classification_done": self._class_done,
            "root_cause_done": self._rc_done,
            "fix_done": self._fix_done,
            "window_start": self._window_start,
            "total_lines": len(self._log_lines),
            "steps_taken": self._steps,
            "max_steps": self._max_steps,
        }
        reward = grade_step(action, self._scenario["ground_truth"], state_snapshot)

        if reward.anomaly_score > 0:
            self._best["anomaly"] = max(self._best["anomaly"], reward.anomaly_score)
            self._anomaly_done = True
        if reward.classification_score > 0:
            self._best["classification"] = max(self._best["classification"], reward.classification_score)
            self._class_done = True
        if reward.root_cause_score > 0:
            self._best["root_cause"] = max(self._best["root_cause"], reward.root_cause_score)
            self._rc_done = True
        if reward.fix_score > 0:
            self._best["fix"] = max(self._best["fix"], reward.fix_score)
            self._fix_done = True
        if reward.penalty == -0.3:  
            self._rc_done = False

        self._steps += 1
        self._cumulative += reward.total

        if reward.done:
            self._done = True
            self.last_result = grade_episode(
                scenario_id=self._scenario["id"],
                task_id=self.task_id,
                best=self._best,
                steps=self._steps,
                max_steps=self._max_steps,
            )

        obs = self._make_obs()
        obs.done = reward.done

        info = {
            "scenario_id": self._scenario["id"],
            "cumulative_reward": round(self._cumulative, 4),
        }
        if reward.done and self.last_result:
            info["episode_result"] = self.last_result.model_dump()

        return obs, reward, reward.done, info

    def state(self):
        return EnvState(
            task_id=self.task_id,
            scenario_id=self._scenario.get("id", "none"),
            steps_taken=self._steps,
            max_steps=self._max_steps,
            done=self._done,
            cumulative_reward=round(self._cumulative, 4),
            anomaly_done=self._anomaly_done,
            classification_done=self._class_done,
            root_cause_done=self._rc_done,
            fix_done=self._fix_done,
        )

    def _apply(self, action: AnalysisAction):           #Update window position or search state based on action."""
        n = len(self._log_lines)
        atype = action.action_type

        if atype == "next_log":
            self._window_start = min(self._window_start + WINDOW_SIZE, max(0, n - WINDOW_SIZE))
        elif atype == "prev_log":
            self._window_start = max(0, self._window_start - WINDOW_SIZE)
        elif atype == "jump_to_index":
            idx = action.target_index or 0
            self._window_start = max(0, min(idx, n - WINDOW_SIZE))
        elif atype == "filter_by_level":
            lvl = action.filter_level
            if lvl:
                # Jump to first line matching the filter level
                for i, line in enumerate(self._log_lines):
                    if f" {lvl} " in line:
                        self._window_start = max(0, min(i, n - WINDOW_SIZE))
                        break
        elif atype == "search_logs":
            kw = (action.search_keyword or "").lower()
            if kw:
                for i, line in enumerate(self._log_lines):
                    if kw in line.lower():
                        self._window_start = max(0, min(i, n - WINDOW_SIZE))
                        break
        # Decision actions (classify, mark_root_cause, suggest_fix) don't move window

    def _make_obs(self):
        start = self._window_start
        end = min(start + WINDOW_SIZE, len(self._log_lines))
        window = [f"[{i:03d}] {self._log_lines[i]}" for i in range(start, end)]
        return LogObservation(
            task_id=self.task_id,
            log_lines=window,
            window_start=start,
            total_lines=len(self._log_lines),
            instruction=self._scenario["instruction"],
            steps_taken=self._steps,
            max_steps=self._max_steps,
            done=self._done,
        )
