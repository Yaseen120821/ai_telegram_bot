"""
app/tools/registry/tool_loader.py - Tool Instantiation & Loading Pipeline
========================================================================
Instantiates discovered BaseTool classes, validates metadata, skips invalid
or broken tools safely, and produces structured ToolRegistration records.
"""

import logging
from typing import List, Type, Dict, Optional, Tuple
from app.tools.base_tool import BaseTool
from app.tools.registry.registry_validator import RegistryValidator
from app.tools.registry.registry_models import ToolRegistration, EnhancedToolMetadata, RegistryReport, RegistryHealth
from app.tools.registry.registry_types import RegistrationStatus, RegistryState
from app.tools.registry.registry_utils import generate_tool_id, format_registry_report
from app.tools.tool_exceptions import ToolException

logger = logging.getLogger("sana_ai.tools.registry.loader")


class ToolLoader:
    """
    Safely instantiates tool classes and validates their registration criteria.
    
    Safety Guarantee:
    - Never invokes core tool execution business logic (`_run`).
    - Catches instantiation and metadata validation errors per-tool.
    - Prevents single broken tools from crashing application startup.
    """

    def __init__(self):
        self.validator = RegistryValidator()

    def load_tool_class(
        self,
        tool_cls: Type[BaseTool],
        existing_tool_names: Dict[str, BaseTool]
    ) -> ToolRegistration:
        """
        Attempts to validate, instantiate, and inspect a BaseTool implementation class.
        """
        cls_name = getattr(tool_cls, '__name__', str(tool_cls))

        # 1. Validate class structure
        try:
            self.validator.validate_tool_class(tool_cls)
        except Exception as val_err:
            logger.warning(f"Class validation failed for '{cls_name}': {val_err}")
            return ToolRegistration(
                tool_id=generate_tool_id(cls_name),
                tool_name=cls_name,
                tool_class=tool_cls,
                status=RegistrationStatus.INVALID_METADATA,
                error_message=str(val_err)
            )

        # 2. Instantiate tool class
        try:
            instance = tool_cls()
        except Exception as inst_err:
            logger.error(f"Failed to instantiate tool class '{cls_name}': {inst_err}", exc_info=True)
            return ToolRegistration(
                tool_id=generate_tool_id(cls_name),
                tool_name=cls_name,
                tool_class=tool_cls,
                status=RegistrationStatus.IMPORT_ERROR,
                error_message=f"Instantiation exception: {inst_err}"
            )

        # 3. Validate instantiated tool & check duplicate names
        try:
            self.validator.validate_tool_instance(instance, existing_tool_names)
        except Exception as dup_or_val_err:
            logger.warning(f"Validation failed for instantiated tool '{instance.name}': {dup_or_val_err}")
            status = RegistrationStatus.DUPLICATE if "Duplicate" in type(dup_or_val_err).__name__ else RegistrationStatus.INVALID_METADATA
            return ToolRegistration(
                tool_id=generate_tool_id(instance.name),
                tool_name=instance.name,
                tool_class=tool_cls,
                instance=instance,
                status=status,
                error_message=str(dup_or_val_err)
            )

        # Successfully loaded
        enhanced_meta = EnhancedToolMetadata(
            name=instance.name,
            description=instance.description,
            category=instance.category,
            version=instance.metadata.version,
            permission_level=instance.permission_level,
            input_schema=instance.parameters_schema,
            timeout_seconds=instance.metadata.timeout_seconds,
            tags=list(instance.metadata.tags),
            state=instance.metadata.state
        )

        return ToolRegistration(
            tool_id=generate_tool_id(instance.name),
            tool_name=instance.name,
            tool_class=tool_cls,
            instance=instance,
            metadata=enhanced_meta,
            status=RegistrationStatus.SUCCESS
        )

    def generate_startup_report(self, registrations: List[ToolRegistration]) -> RegistryReport:
        """Constructs a comprehensive audit report of loaded tools."""
        total_scanned = len(registrations)
        total_success = sum(1 for r in registrations if r.status == RegistrationStatus.SUCCESS)
        total_failed = total_scanned - total_success

        errors = [f"Tool '{r.tool_name}': {r.error_message}" for r in registrations if r.status != RegistrationStatus.SUCCESS]
        is_healthy = len(errors) == 0

        health = RegistryHealth(
            state=RegistryState.READY if is_healthy else RegistryState.ERROR,
            is_healthy=is_healthy,
            loaded_tools_count=total_success,
            failed_imports_count=total_failed,
            errors=errors
        )

        report = RegistryReport(
            total_scanned=total_scanned,
            total_registered=total_success,
            total_failed=total_failed,
            registrations=registrations,
            health=health
        )

        logger.info(format_registry_report(report))
        return report
