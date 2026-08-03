"""
app/tools/permission_manager.py - Authorization & Security Governance
====================================================================
Evaluates security policies, user roles, permission levels, and confirmation flags
to prevent dangerous or unauthorized tool calls by the AI system.
"""

import logging
from typing import Dict, Any, Optional, Set
from app.tools.tool_types import PermissionLevel
from app.tools.tool_models import ToolMetadata, PermissionResult, ExecutionContext
from app.tools.tool_config import get_tool_config

logger = logging.getLogger("sana_ai.tools.permission")


class PermissionManager:
    """
    Manages execution permissions for all tools.
    
    Security Guarantee:
    Ensures that high-risk or administrative operations require human confirmation
    or explicit administrative permissions before being dispatched for execution.
    """

    def __init__(self):
        self.config = get_tool_config()
        self.auto_approved_tools: Set[str] = set()
        self.blacklisted_tools: Set[str] = set()

    def check_permission(
        self,
        tool_metadata: ToolMetadata,
        user_confirmed: bool = False,
        context: Optional[ExecutionContext] = None
    ) -> PermissionResult:
        """
        Evaluates whether a tool request is authorized to execute.
        
        Rules:
        1. If blacklisted -> DENIED
        2. SAFE level -> ALLOWED automatically (unless auto-approve disabled)
        3. CONFIRMATION_REQUIRED -> ALLOWED ONLY IF `user_confirmed` is True
        4. RESTRICTED -> Requires user to have 'restricted' permission or explicit admin role
        5. ADMINISTRATOR -> Requires 'admin' role in context and admin_mode_enabled in config
        """
        tool_name = tool_metadata.name
        perm_level = tool_metadata.permission_level

        # Check explicit blacklist
        if tool_name in self.blacklisted_tools:
            logger.warning(f"Permission denied for blacklisted tool '{tool_name}'.")
            return PermissionResult(
                is_allowed=False,
                permission_level=perm_level,
                requires_confirmation=False,
                reason=f"Tool '{tool_name}' is explicitly blacklisted."
            )

        # Explicit whitelist override
        if tool_name in self.auto_approved_tools:
            return PermissionResult(
                is_allowed=True,
                permission_level=perm_level,
                requires_confirmation=False,
                reason=f"Tool '{tool_name}' is whitelisted for auto-approval."
            )

        # Rule 1: SAFE tools
        if perm_level == PermissionLevel.SAFE:
            return PermissionResult(
                is_allowed=True,
                permission_level=perm_level,
                requires_confirmation=False,
                reason="SAFE tool execution granted automatically."
            )

        # Rule 2: CONFIRMATION_REQUIRED tools
        if perm_level == PermissionLevel.CONFIRMATION_REQUIRED:
            if user_confirmed:
                return PermissionResult(
                    is_allowed=True,
                    permission_level=perm_level,
                    requires_confirmation=True,
                    user_confirmed=True,
                    reason="User explicitly confirmed operation."
                )
            else:
                logger.info(f"Tool '{tool_name}' requires explicit user confirmation.")
                return PermissionResult(
                    is_allowed=False,
                    permission_level=perm_level,
                    requires_confirmation=True,
                    user_confirmed=False,
                    reason="Execution blocked: Requires explicit user confirmation."
                )

        # Rule 3: RESTRICTED tools
        if perm_level == PermissionLevel.RESTRICTED:
            has_restricted = context and ("restricted" in context.user_permissions or "admin" in context.user_permissions)
            if has_restricted and user_confirmed:
                return PermissionResult(
                    is_allowed=True,
                    permission_level=perm_level,
                    requires_confirmation=True,
                    user_confirmed=True,
                    reason="RESTRICTED privilege & confirmation verified."
                )
            elif not has_restricted:
                return PermissionResult(
                    is_allowed=False,
                    permission_level=perm_level,
                    requires_confirmation=True,
                    reason="Execution blocked: Missing 'restricted' privilege in user context."
                )
            else:
                return PermissionResult(
                    is_allowed=False,
                    permission_level=perm_level,
                    requires_confirmation=True,
                    reason="Execution blocked: RESTRICTED tool requires user confirmation."
                )

        # Rule 4: ADMINISTRATOR tools
        if perm_level == PermissionLevel.ADMINISTRATOR:
            is_admin = (self.config.admin_mode_enabled or (context and "admin" in context.user_permissions))
            if is_admin and user_confirmed:
                return PermissionResult(
                    is_allowed=True,
                    permission_level=perm_level,
                    requires_confirmation=True,
                    user_confirmed=True,
                    reason="ADMINISTRATOR level access verified."
                )
            else:
                return PermissionResult(
                    is_allowed=False,
                    permission_level=perm_level,
                    requires_confirmation=True,
                    reason="Execution blocked: Administrator privileges and explicit confirmation required."
                )

        return PermissionResult(
            is_allowed=False,
            permission_level=perm_level,
            reason="Unknown security level encountered."
        )

    def whitelist_tool(self, tool_name: str) -> None:
        """Adds a tool to auto-approval list."""
        self.auto_approved_tools.add(tool_name)

    def blacklist_tool(self, tool_name: str) -> None:
        """Blocks tool from execution regardless of permission level."""
        self.blacklisted_tools.add(tool_name)
