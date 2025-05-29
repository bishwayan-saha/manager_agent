import ast
import logging
import os
from typing import List

import requests

from models.agent import AgentCard

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiscoveryClient:
    """
    Discover A2A agents by reading a registry file of agent server URLs and querying
    each one's /.well-known/agent.json endpoint to retrieve an AgentCard.

    Attributes:
        registry_path (str): The path to the registry file containing a list of agent server URLs.
        base_urls (list[str]): A list of agent server URLs to query.
    """

    # def __init__(self, registry_path: str):
    #     if registry_path:
    #         self.registry_path = registry_path
    #     else:
    #         self.registry_path = os.path.join(
    #             os.path.dirname(__file__), "registry.json"
    #         )

    async def fetch_agent_cards(self) -> List[AgentCard]:
        """
        Asynchronously fetch the discovery endpoint from each registered URL
        and parse the returned JSON into AgentCard objects.

        Returns:
            List[AgentCard]: Successfully retrieved agent cards.
        """
        agent_cards: List[AgentCard] = []
        responses = requests.get(
            "https://interop-ae-chat.azurewebsites.net/agent_cards"
        ).json()

        for response in responses["data"]:
            try:
                card = AgentCard.model_validate(response)
                agent_cards.append(card)
            except Exception as e:
                logger.info(f"Error occurred while fetching well known url {e}")

        return agent_cards
