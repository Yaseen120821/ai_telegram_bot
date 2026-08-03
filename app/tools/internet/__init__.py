"""
app/tools/internet package initializer - External Web & Internet Tools Architecture
===================================================================================
Defines architectural placeholders and contract declarations for future live web APIs:
Weather, News, Wikipedia, Currency Exchange, Search Engine, Maps, Email, Calendar,
and Headless Browser automation tools.
"""

from typing import Dict, Any, Optional
from app.tools.base_tool import BaseTool
from app.tools.tool_types import ToolCategory, PermissionLevel


class ExternalWebAPIPlaceholder(BaseTool):
    """Base class architectural placeholder for future live external web APIs."""
    def __init__(self, name: str, description: str, endpoint: str):
        super().__init__(
            name=name,
            description=description,
            category=ToolCategory.INTERNET,
            permission_level=PermissionLevel.RESTRICTED,
            parameters_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            tags=["internet", "web", "external_api", name]
        )
        self.endpoint = endpoint

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        return {
            "status": "placeholder",
            "message": f"External API tool '{self.name}' [Endpoint: {self.endpoint}] is architecturally declared for future integration.",
            "parameters": parameters
        }


__all__ = ["ExternalWebAPIPlaceholder"]
