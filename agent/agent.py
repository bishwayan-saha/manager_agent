import logging
import os
import uuid
from typing import List

import requests
from dotenv import load_dotenv
from google.adk.agents.llm_agent import LlmAgent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from agent.agent_connector import AgentConnector
from mcp_connect import MCPConnector
from models.agent import AgentCard

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HostAgent:
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self, agent_cards: List[AgentCard]) -> None:
        self.agent_connectors = {
            card.name: AgentConnector(card.name, card.url) for card in agent_cards
        }

        self.agent_descriptions = {
            card.name: f"{card.description + " " + ".\n ".join([skill.description for skill in card.skills])}" for card in agent_cards
        }
        server_domain = os.getenv("SERVER_DOMAIN") or "http://localhost"
        logger.info(f"Server domain for calling credentials: {server_domain}")
        self._credentials = requests.get(
            f"{server_domain}:3100/credentials"
        ).json()
        for creds in self._credentials["data"]:
            os.environ[creds] = self._credentials["data"].get(creds)

        load_dotenv()
        self._mcp = MCPConnector()
        mcp_tools = self._mcp.get_tools()

        self._mcp_wrappers = []

        def make_wrapper(tool):
            async def wrapper(args: dict) -> str:
                return await tool.run(args)

            wrapper.__name__ = tool.name
            return wrapper

        for tool in mcp_tools:
            fn = make_wrapper(tool)
            self._mcp_wrappers.append(FunctionTool(fn))

        self._agent = self._build_agent()
        self._user_id = "host_agent"

        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

    def _build_agent(self) -> LlmAgent:
        return LlmAgent(
            model="gemini-2.0-flash",
            name="host_agent",
            description="""The host agent is responsible for coordinating the 
                           actions of the other agents based on user query intent.
                           """,
            instruction=f"""
                            ## Host Manager Agent Instructions
                            ### **Role & Purpose**
                            You are a **host manager agent** responsible for managing tasks and coordinating with other agents based on user intent.
                            ### **Tool Categories**
                            1. **A2A Agent Tools**:
                            - `list_agents()`: Retrieve a list of available agents.
                            - `delegate_task(agent_name, message)`: Assign tasks to an agent.

                            2. **MCP Tools**:
                            - Function tools, including an **Airbnb MCP tool** Use this tool only and no other tools or agents for any hotel related query

                            ### **Available Remote Agents**
                            - {", ".join([agent for agent in self.agent_connectors.keys()])}
                            - {str(self.agent_descriptions)}
                            - If you are certain about using an remote agent based on user prompt, delegate the task immediately.
                            - While using remote agent, **DO NOT** generate response by yourself, 
                                wait for the remote agent response and pass that to the caller

                            ### **Guidelines**
                            - **Use available tools** to fetch or process information instead of guessing.
                            - **Avoid assumptions**—never generate responses based on unknown data.
                            - **Ask for clarification** when needed.
                            - **Provide structured, clear, and helpful responses**.
                            - **Communicate capabilities effectively** while assisting users.
                        """,
                        
            tools=[self._list_agents, self._delegate_task, *self._mcp_wrappers],
            output_key="manager"
        )

    def _list_agents(self) -> List[str]:
        """
        Tool function: returns the list of child-agent names currently registered.
        Called by the LLM when it wants to discover available agents.
        """
        print(f"List agents {list(self.agent_connectors.keys())}")
        return list(self.agent_connectors.keys())

    async def _delegate_task(
        self, agent_name: str, message: str, tool_context: ToolContext
    ) -> str:
        """
        Tool function: Delegate a task to an agent.
        """
        if agent_name not in self.agent_connectors:
            raise ValueError(f"Unknown agent: {agent_name}")
        connector = self.agent_connectors[agent_name]

        # Ensure session_id persists across tool calls via tool_context.state
        state = tool_context.state
        if "session_id" not in state:
            state["session_id"] = str(uuid.uuid4())
        session_id = state["session_id"]

        # Delegate task asynchronously and await Task result
        child_task = await connector.send_task(message, session_id)

        # Extract text from the last history entry if available
        if child_task.history and len(child_task.history) > 1:
            return child_task.history[-1].parts[0].text
        return ""

    def invoke(self, query: str, session_id: str) -> str:
        """
        Main entry: receives a user query + session_id,
        sets up or retrieves a session, wraps the query for the LLM,
        runs the Runner (with tools enabled), and returns the final text.
        """
        # Attempt to reuse an existing session
        session = self._runner.session_service.get_session(
            app_name=self._agent.name, user_id=self._user_id, session_id=session_id
        )
        # Create new if not found
        if session is None:
            session = self._runner.session_service.create_session(
                app_name=self._agent.name,
                user_id=self._user_id,
                session_id=session_id,
                state={},
            )

        # Wrap the user query in a types.Content message
        content = types.Content(role="user", parts=[types.Part.from_text(text=query)])

        # Run the agent synchronously; collects a list of events
        events = list(
            self._runner.run(
                user_id=self._user_id, session_id=session.id, new_message=content
            )
        )
        current_session = self._runner.session_service.get_session(app_name=self._agent.name, user_id=self._user_id, session_id=session_id)
        stored_output = current_session.state.get(self._runner.agent.output_key)

        # If no content or parts, return empty fallback
        if not events or not events[-1].content or not events[-1].content.parts:
            return ""
        # Join all text parts into a single string reply
        return "\n".join(p.text for p in events[-1].content.parts if p.text)
