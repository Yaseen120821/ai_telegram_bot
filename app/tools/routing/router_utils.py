"""
app/tools/routing/router_utils.py - Parameter Parsing & Pattern Utilities
========================================================================
Provides regex helpers, parameter extraction utilities for file paths/expressions,
and text normalization functions used by the routing engine.
"""

import re
from typing import Dict, Any, Optional, Tuple


def normalize_query_text(query: str) -> str:
    """Cleans and normalizes query text for pattern matching."""
    text = query.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    return text


def extract_filepath_from_text(query: str) -> Optional[str]:
    """
    Extracts filename or file path patterns from natural language queries.
    Examples:
      - 'read report.pdf' -> 'report.pdf'
      - 'read my file notes.txt' -> 'notes.txt'
    """
    # Regex 1: Matches filenames with explicit extension (.txt, .pdf, .py, .doc, etc.)
    match_ext = re.search(r'([a-zA-Z0-9_\-\/\\]+\.[a-zA-Z0-9]{2,4})', query)
    if match_ext:
        return match_ext.group(1)

    # Regex 2: Matches "read/open/cat [file] <name>"
    match_keyword = re.search(r'(?:read|open|view|show|cat)\s+(?:file|document|my)?\s*([a-zA-Z0-9_\-\.]+)', query, re.IGNORECASE)
    if match_keyword:
        fname = match_keyword.group(1).strip()
        if fname not in ("file", "document", "my", "the", "a"):
            return fname

    return None


def extract_arithmetic_expression(query: str) -> Optional[str]:
    """
    Extracts math expression string from natural language calculation queries.
    Examples:
      - 'calculate 15 * 28' -> '15 * 28'
      - 'what is 100 / 5' -> '100 / 5'
    """
    # Look for math operators between numbers
    match = re.search(r'([\d\.\s\+\-\*\/\(\)]+[\+\-\*\/\(\)][\d\.\s\+\-\*\/\(\)]+)', query)
    if match:
        expr = match.group(1).strip()
        # Clean up trailing punctuation
        expr = re.sub(r'[^\d\.\+\-\*\/\(\)\s]', '', expr)
        if expr:
            return expr

    return None
