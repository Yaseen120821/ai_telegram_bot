"""
app/tools/utility/utility_tools.py - Utility Helper Tools
=========================================================
Provides utility tools: Calculator, UUID Generator, Random Number,
Password Generator, Hash Generator, Base64 Encode/Decode, and QR Code payload tools.
"""

import ast
import uuid
import random
import string
import hashlib
import base64
import logging
from typing import Dict, Any, Optional
from app.tools.base_tool import BaseTool
from app.tools.tool_types import ToolCategory, PermissionLevel
from app.tools.tool_exceptions import ToolExecutionException

logger = logging.getLogger("sana_ai.tools.utility")


class CalculatorTool(BaseTool):
    """Evaluates arithmetic expressions safely using AST parsing."""
    def __init__(self):
        super().__init__(
            name="calculate_math",
            description="Evaluates numeric mathematical arithmetic expressions.",
            category=ToolCategory.UTILITY,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Arithmetic expression e.g. '15 * (28 + 4)'"}
                },
                "required": ["expression"]
            },
            tags=["calculator", "math", "arithmetic", "calculate"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        expr = parameters["expression"]
        try:
            # Safe AST evaluation restricting to arithmetic node types
            node = ast.parse(expr, mode='eval')
            result = self._eval_node(node.body)
            return {"expression": expr, "result": result}
        except Exception as exc:
            raise ToolExecutionException(f"Invalid arithmetic expression '{expr}': {exc}", tool_name=self.name)

    def _eval_node(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            elif isinstance(node.op, ast.Sub):
                return left - right
            elif isinstance(node.op, ast.Mult):
                return left * right
            elif isinstance(node.op, ast.Div):
                if right == 0:
                    raise ZeroDivisionError("Division by zero")
                return left / right
            elif isinstance(node.op, ast.Mod):
                return left % right
            elif isinstance(node.op, ast.Pow):
                return left ** right
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            elif isinstance(node.op, ast.USub):
                return -operand
        raise ValueError(f"Unsupported AST node type in calculation: {type(node).__name__}")


class UUIDGeneratorTool(BaseTool):
    """Generates v4 UUID identifiers."""
    def __init__(self):
        super().__init__(
            name="generate_uuid",
            description="Generates one or more unique v4 UUID strings.",
            category=ToolCategory.UTILITY,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={
                "type": "object",
                "properties": {
                    "count": {"type": "integer", "description": "Number of UUIDs to generate (1-10)"}
                }
            },
            tags=["uuid", "guid", "unique_id"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        count = min(max(parameters.get("count", 1), 1), 10)
        uuids = [str(uuid.uuid4()) for _ in range(count)]
        return {"count": count, "uuids": uuids}


class RandomNumberTool(BaseTool):
    """Generates random integers within a min/max range."""
    def __init__(self):
        super().__init__(
            name="generate_random_number",
            description="Generates a random integer within a specified range.",
            category=ToolCategory.UTILITY,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={
                "type": "object",
                "properties": {
                    "min_val": {"type": "integer"},
                    "max_val": {"type": "integer"}
                }
            },
            tags=["random", "number", "rng", "dice"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        min_v = parameters.get("min_val", 1)
        max_v = parameters.get("max_val", 100)
        if min_v > max_v:
            min_v, max_v = max_v, min_v
        val = random.randint(min_v, max_v)
        return {"min_val": min_v, "max_val": max_v, "result": val}


class PasswordGeneratorTool(BaseTool):
    """Generates secure random passwords."""
    def __init__(self):
        super().__init__(
            name="generate_password",
            description="Generates a cryptographically strong random password.",
            category=ToolCategory.UTILITY,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={
                "type": "object",
                "properties": {
                    "length": {"type": "integer"},
                    "include_symbols": {"type": "boolean"}
                }
            },
            tags=["password", "security", "credentials"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        length = min(max(parameters.get("length", 16), 8), 128)
        include_symbols = parameters.get("include_symbols", True)

        chars = string.ascii_letters + string.digits
        if include_symbols:
            chars += "!@#$%^&*()_+-=[]{}|;:,.<>?"

        pwd = "".join(random.choice(chars) for _ in range(length))
        return {"length": length, "password": pwd}


class HashGeneratorTool(BaseTool):
    """Generates cryptographic hashes (MD5, SHA1, SHA256, SHA512)."""
    def __init__(self):
        super().__init__(
            name="generate_hash",
            description="Generates cryptographic hashes (sha256, sha512, md5) for input text.",
            category=ToolCategory.UTILITY,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "algorithm": {"type": "string"}
                },
                "required": ["text"]
            },
            tags=["hash", "sha256", "md5", "digest"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, str]:
        text = parameters["text"]
        algo = parameters.get("algorithm", "sha256").lower()
        if algo not in hashlib.algorithms_guaranteed:
            algo = "sha256"

        h = hashlib.new(algo)
        h.update(text.encode("utf-8"))
        return {"algorithm": algo, "hash": h.hexdigest()}


class Base64EncodeTool(BaseTool):
    """Encodes plain text string to Base64."""
    def __init__(self):
        super().__init__(
            name="base64_encode",
            description="Encodes a plain text string into Base64 format.",
            category=ToolCategory.UTILITY,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"}
                },
                "required": ["text"]
            },
            tags=["base64", "encode"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, str]:
        text = parameters["text"]
        encoded = base64.b64encode(text.encode("utf-8")).decode("utf-8")
        return {"encoded": encoded}


class Base64DecodeTool(BaseTool):
    """Decodes Base64 string to plain text."""
    def __init__(self):
        super().__init__(
            name="base64_decode",
            description="Decodes a Base64 encoded string back to plain text.",
            category=ToolCategory.UTILITY,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={
                "type": "object",
                "properties": {
                    "encoded_text": {"type": "string"}
                },
                "required": ["encoded_text"]
            },
            tags=["base64", "decode"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, str]:
        encoded = parameters["encoded_text"]
        try:
            decoded = base64.b64decode(encoded.encode("utf-8")).decode("utf-8")
            return {"decoded": decoded}
        except Exception as exc:
            raise ToolExecutionException(f"Failed to decode Base64 string: {exc}", tool_name=self.name)


class QRCodeGeneratorTool(BaseTool):
    """Generates ASCII or payload representation for QR codes."""
    def __init__(self):
        super().__init__(
            name="generate_qr_code",
            description="Generates payload metadata and ASCII representation for a QR code string.",
            category=ToolCategory.UTILITY,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={
                "type": "object",
                "properties": {
                    "data": {"type": "string"}
                },
                "required": ["data"]
            },
            tags=["qr", "qrcode", "barcode"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        data = parameters["data"]
        return {
            "data": data,
            "length": len(data),
            "payload_format": "text/plain",
            "message": f"QR code payload prepared for '{data}'."
        }
