import asyncio
import logging

import click

from agent.agent import HostAgent
from agent.task_manager import HostAgentTaskManager
from discovery import DiscoveryClient
from models.agent import AgentCapabilities, AgentCard, AgentSkill
from server.server import A2AServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--host", default="localhost", help="Host to bind the HostAgent server to"
)
@click.option("--port", default=10000, help="Port for the HostAgent server")
# @click.option(
#     "--registry",
#     default=None,
#     help=(
#         "Path to JSON file listing child-agent URLs. "
#         "Defaults to registry.json"
#     )
# )
def main(host: str, port: int):
    """
    Entry point to start the OrchestratorAgent A2A server.

    Steps performed:
    1. Load child-agent URLs from the registry JSON file.
    2. Fetch each agent's metadata via `/.well-known/agent.json`.
    3. Instantiate an OrchestratorAgent with discovered AgentCards.
    4. Wrap it in an OrchestratorTaskManager for JSON-RPC handling.
    5. Launch the A2AServer to listen for incoming tasks.
    """
    logger.info(" --- Host Agent Started --- ")
    discovery = DiscoveryClient()
    agent_cards = asyncio.run(discovery.fetch_agent_cards())

    if not agent_cards:
        logger.warning(
            "No agents found in registry – the orchestrator will have nothing to call"
        )
    capabilities = AgentCapabilities(streaming=False)
    skill = AgentSkill(
        id="orchestrate_agents",
        name="Orchestrate Agents and Tasks",
        description=(
            "Routes user requests to the appropriate child agent, "
            "based on intent of the user query"
        ),
        tags=["routing", "orchestration"],
        examples=[],
    )
    host_agent_card = AgentCard(
        name="HostAgent",
        description="Delegates tasks to discovered child agents",
        url=f"http://{host}:{port}/",  #
        version="1.0.0",
        defaultInputModes=HostAgent.SUPPORTED_CONTENT_TYPES,
        defaultOutputModes=HostAgent.SUPPORTED_CONTENT_TYPES,
        capabilities=capabilities,
        skills=[skill],
    )
    host_agent = HostAgent(agent_cards=agent_cards)
    task_manager = HostAgentTaskManager(agent=host_agent)
    server = A2AServer(
        host=host, port=port, agent_card=host_agent_card, task_manager=task_manager
    )
    server.start()


if __name__ == "__main__":
    main()
