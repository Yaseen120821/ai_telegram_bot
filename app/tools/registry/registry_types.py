"""
app/tools/registry/registry_types.py - Enumeration Types for SANA AI Tool Registry
=============================================================================
Defines standard enums for registry lifecycle states, registration outcomes,
discovery modes, tool visibility, and registration error classifications.
"""

from enum import Enum


class RegistryState(str, Enum):
    """Lifecycle states of the Tool Registry system."""
    UNINITIALIZED = "uninitialized"
    DISCOVERING = "discovering"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class RegistrationStatus(str, Enum):
    """Status outcomes for tool registration attempts."""
    SUCCESS = "success"
    DUPLICATE = "duplicate"
    INVALID_METADATA = "invalid_metadata"
    MISSING_BASE_CLASS = "missing_base_class"
    IMPORT_ERROR = "import_error"
    DISABLED = "disabled"
    SKIPPED = "skipped"


class DiscoveryMode(str, Enum):
    """Modes determining how tools are discovered at application startup."""
    STATIC = "static"           # Manually registered tool classes only
    DYNAMIC_AUTO = "dynamic_auto" # Automatically scans directories for BaseTool subclasses
    HYBRID = "hybrid"           # Combines static pre-registered list with dynamic folder scanning
    EXPLICIT = "explicit"       # Manifest-driven explicit registration list


class ToolVisibility(str, Enum):
    """Visibility scoping for registered tools."""
    PUBLIC = "public"           # Exposed to standard user intent router and LLM prompt schemas
    INTERNAL = "internal"       # System-only internal utility tools
    HIDDEN = "hidden"           # Hidden from general LLM prompt schemas (callable by name only)
    EXPERIMENTAL = "experimental"# Feature-flagged tools under evaluation


class RegistryError(str, Enum):
    """Specific classification codes for registry operation failures."""
    NONE = "none"
    DUPLICATE_NAME = "duplicate_name"
    INVALID_SCHEMA = "invalid_schema"
    IMPORT_FAILED = "import_failed"
    TYPE_MISMATCH = "type_mismatch"
    NOT_FOUND = "not_found"
    PERMISSION_ERROR = "permission_error"
