import logging
import uuid

from client.client import A2AClient
from models.task import Task

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentConnector:
    """
    Connects to a remote A2A agent and provides a uniform method to delgates the tasks
    Attributes:
        name (str): remote agent identifier name
        client (A2AClient): HTTP Cliet pointing to Agent's server URL
    """

    def __init__(self, name: str, base_url: str) -> None:
        self.name = name
        self.client = A2AClient(url=base_url)
        logger.info(f"AgentConnector initialized for {name} at {base_url}")

    async def send_task(self, message: str, session_id: str) -> Task:
        """ """
        task_id = uuid.uuid4().hex
        payload = {
            "id": task_id,
            "sessionId": session_id,
            "message": {"role": "user", "parts": [{"type": "text", "text": message}]},
        }
        task_result = await self.client.send_task(payload)
        logger.info(
            f"AgentConnector: received response from {self.name} for task {task_id}"
        )
        return task_result
