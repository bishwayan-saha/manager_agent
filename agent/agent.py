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


class HostAgent:
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self, agent_cards: List[AgentCard]) -> None:
        self.agent_connectors = {
            card.name: AgentConnector(card.name, card.url) for card in agent_cards
        }

        self._credentials = requests.get(
            "https://interop-ae-chat.azurewebsites.net/credentials"
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
                            You are a host manager agent with specialized tools to assist you in managing tasks and coordinating with other agents. 
                            Your primary function is to understand what the user needs and utilize your available tools to provide accurate responses.
                           
                           ### You have two categories of tools available to you
                            1. A2A Agent tools: There are some tools under this category
                                1.1-> list_agents(): List all the agents available to you.
                                1.2-> delegate_task(agent_name, message): Delegate a task to an agent.
                            2. MCP tools: one FuntionTool per tool name
                                2.1-> This is an airbnb mcp tool

                            ### Available agents (separated by comma): 
                            {", ".join([agent for agent in self.agent_connectors.keys()])}

                            ### Guidelines:
                                - Always rely on the tools and agents provided to gather information or complete tasks.  
                                -  Do **not** make assumptions or "hallucinate" and never assume or generate responses based on 
                                    information you don’t have.  
                                - If you need more details to respond accurately, ask the user for clarification.  
                                - Use structured and clear responses, making sure they are helpful
                                - When interacting with users, explain your capabilities clearly and assist them in a structured manner
                        """,
            tools=[self._list_agents, self._delegate_task, *self._mcp_wrappers],
        )

    def _list_agents(self) -> List[str]:
        """
        Tool function: returns the list of child-agent names currently registered.
        Called by the LLM when it wants to discover available agents.
        """
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

        # If no content or parts, return empty fallback
        if not events or not events[-1].content or not events[-1].content.parts:
            return ""
        # Join all text parts into a single string reply
        return "\n".join(p.text for p in events[-1].content.parts if p.text)
