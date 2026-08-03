"""
app/tools/execution/permission_checker.py - Permission Governance Engine
========================================================================
Evaluates security policies, authorization levels, user roles, blacklists,
and human confirmation flags before allowing tool execution.
"""

import logging
from typing import Optional, Set, Tuple
from app.tools.base_tool import BaseTool
from app.tools.tool_types import PermissionLevel
from app.tools.execution.execution_types import PermissionDecision
from app.tools.execution.execution_models import ExecutionContext
from app.tools.execution.execution_config import get_execution_config

logger = logging.getLogger("sana_ai.tools.execution.permission")


class PermissionChecker:
    """
    Security Governance Layer.
    
    Security Guarantees:
    - Current Time (SAFE) -> Allowed automatically.
    - Delete File (CONFIRMATION_REQUIRED) -> Requires explicit user confirmation.
    - Execute Shell (RESTRICTED) -> Requires user confirmation & restricted role.
    - System Shutdown (ADMINISTRATOR) -> Requires user confirmation & admin role.
    """

    def __init__(self):
        self.config = get_execution_config()
        self.blacklisted_tools: Set[str] = set()
        self.whitelisted_tools: Set[str] = set()

    def evaluate_permission(
        self,
        tool: BaseTool,
        user_confirmed: bool = False,
        context: Optional[ExecutionContext] = None
    ) -> Tuple[PermissionDecision, str]:
        """
        Evaluates whether a tool request is authorized to execute.
        Returns Tuple[PermissionDecision, reason_explanation].
        """
        tool_name = tool.name
        perm_level = tool.permission_level

        # 1. Blacklist check
        if tool_name in self.blacklisted_tools:
            logger.warning(f"Permission denied: Tool '{tool_name}' is blacklisted.")
            return PermissionDecision.DENIED_BLACK_LISTED, f"Tool '{tool_name}' is explicitly blacklisted."

        # 2. Whitelist override or SAFE level
        if tool_name in self.whitelisted_tools or perm_level == PermissionLevel.SAFE:
            return PermissionDecision.ALLOWED, "Permission granted automatically."

        # 3. CONFIRMATION_REQUIRED
        if perm_level == PermissionLevel.CONFIRMATION_REQUIRED:
            if user_confirmed:
                return PermissionDecision.ALLOWED, "User explicitly confirmed high-risk operation."
            logger.info(f"Permission blocked: Tool '{tool_name}' requires user confirmation.")
            return PermissionDecision.DENIED_CONFIRMATION_REQUIRED, f"Tool '{tool_name}' requires explicit user confirmation."

        # 4. RESTRICTED
        if perm_level == PermissionLevel.RESTRICTED:
            user_perms = context.env_vars.get("user_permissions", "").split(",") if context else []
            has_privilege = "restricted" in user_perms or "admin" in user_perms
            if has_privilege and user_confirmed:
                return PermissionDecision.ALLOWED, "RESTRICTED privileges and user confirmation verified."
            elif not has_privilege:
                return PermissionDecision.DENIED_INSUFFICIENT_ROLE, "Missing 'restricted' privilege in user context."
            else:
                return PermissionDecision.DENIED_CONFIRMATION_REQUIRED, "RESTRICTED operations require user confirmation."

        # 5. ADMINISTRATOR
        if perm_level == PermissionLevel.ADMINISTRATOR:
            user_perms = context.env_vars.get("user_permissions", "").split(",") if context else []
            is_admin = "admin" in user_perms or context.user_id == "admin"
            if is_admin and user_confirmed:
                return PermissionDecision.ALLOWED, "ADMINISTRATOR access and user confirmation verified."
            return PermissionDecision.DENIED_INSUFFICIENT_ROLE, "ADMINISTRATOR privilege and user confirmation required."

        return PermissionDecision.DENIED_INSUFFICIENT_ROLE, "Unknown security permission level."
