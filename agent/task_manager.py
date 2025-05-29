import logging

from agent.agent import HostAgent
from models.request import SendTaskRequest, SendTaskResponse
from models.task import Message, TaskState, TaskStatus, TextPart
from server.task_manager import InMemoryTaskManager
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HostAgentTaskManager(InMemoryTaskManager):
    """
    🪄 TaskManager wrapper: exposes HostAgent.invoke() over the
    A2A JSON-RPC `tasks/send` endpoint, handling in-memory storage and
    response formatting.
    """

    def __init__(self, agent: HostAgent):
        super().__init__()  # Initialize base in-memory storage
        self.agent = agent  # Store our orchestrator logic

    def _get_user_text(self, request: SendTaskRequest) -> str:
        """
        Helper: extract the user's raw input text from the request object.
        """
        return request.params.message.parts[0].text

    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        """
        Called by the A2A server when a new task arrives:
        1. Store the incoming user message
        2. Invoke the HostAgent to get a response
        3. Append response to history, mark completed
        4. Return a SendTaskResponse with the full Task
        """
        logger.info(f"OrchestratorTaskManager received task {request.params.id}")

        # Step 1: save the initial message
        task = await self.upsert_task(request.params)

        # Step 2: run orchestration logic
        user_text = self._get_user_text(request)
        response_text = self.agent.invoke(user_text, request.params.session_id)

        # Step 3: wrap the LLM output into a Message
        reply = Message(role="agent", parts=[TextPart(text=response_text)])
        logger.info(f"\nOutgoing JSON Response:\n {json.dumps(reply.model_dump(), indent=2)}")
        async with self.lock:
            task.status = TaskStatus(state=TaskState.COMPLETED)
            task.history.append(reply)
            print(f"\n <<<<<<<<<<<  TASK HISTORY >>>>>>>>>>>>>\n {task.history}")

        # Step 4: return structured response
        return SendTaskResponse(id=request.id, result=task)
