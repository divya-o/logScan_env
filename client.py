from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import LogscanAction, LogscanObservation


class LogscanEnv(
    EnvClient[LogscanAction, LogscanObservation, State]
):
    """
    Example:
        >>> # Connect to a running server
        >>> with LogscanEnv(base_url="http://localhost:8000") as client:
        ...     result=client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result=client.step(LogscanAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client=LogscanEnv.from_docker_image("logScan_env-env:latest")
        >>> try:
        ...     result=client.reset()
        ...     result=client.step(LogscanAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: LogscanAction) -> Dict:
       
        return {
            "message": action.message,
        }

    def _parse_result(self, payload: Dict) -> StepResult[LogscanObservation]:
     
        obs_data=payload.get("observation", {})
        observation=LogscanObservation(
            echoed_message=obs_data.get("echoed_message", ""),
            message_length=obs_data.get("message_length", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
