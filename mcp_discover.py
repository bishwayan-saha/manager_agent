import json
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPToolDiscovery:

    def __init__(self, config_file: str) -> None:

        if config_file:
            self._config_file = config_file
        else:
            self._config_file = os.path.join(
                os.path.dirname(__file__), "mcp_config.json"
            )
        self._config = self._load_config()

    def _load_config(self):
        try:
            with open(self._config_file, "r") as file:
                data = json.load(file)
            return data
        except Exception as e:
            logger.error(f" Error occurred while geeting mcp config. \n Reason: {e}")

    def list_servers(self):
        logger.info(f" MCOP Servers{self._config.get("mcpServers", {})}")
        return self._config.get("mcpServers", {})
