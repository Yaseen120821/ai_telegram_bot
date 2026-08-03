"""
app/memory/memory_types.py - Memory Categories & Importance Enumerations
==========================================================================

1. PURPOSE:
-----------
Defines strongly-typed Enumeration classes (`MemoryCategory` and `ImportanceLevel`) for categorizing
stored facts and assigning retrieval priority weights (1 to 10 scale).

2. WHY IT EXISTS:
-----------------
Using raw string literals (like `"profile"`, `"preference"`) across multiple files causes typos and schema drift.
Enums provide autocomplete support, type checking, and central definition for valid memory taxonomy categories.

3. RESPONSIBILITIES:
--------------------
- Define valid taxonomy categories (`profile`, `preference`, `goal`, `project`, `skill`, `education`, `career`,
  `relationship`, `achievement`, `interest`, `custom`).
- Define importance weighting scale (1 to 10).

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Used by `app/memory/memory_models.py`, `app/memory/memory_classifier.py`, `app/memory/memory_store.py`,
  `app/memory/memory_retriever.py`, and `app/memory/memory_utils.py`.

5. COMPLETE CODE:
-----------------
"""

from enum import Enum


class MemoryCategory(str, Enum):
    """
    Standard taxonomy categories for personal long-term memory classification.
    """
    PROFILE = "profile"            # Identity facts (Name, Age, Location, Birthday)
    PREFERENCE = "preference"      # Likes, Dislikes, Coding language preferences
    GOAL = "goal"                  # Career, Learning, and Personal targets
    PROJECT = "project"            # Active software endeavors (e.g. SANA AI)
    SKILL = "skill"                # Technical & Professional competencies
    EDUCATION = "education"        # Degree, School, Major, Field of study
    CAREER = "career"              # Job title, Company, Industry
    RELATIONSHIP = "relationship"  # Friends, Family, Mentor names
    ACHIEVEMENT = "achievement"    # Milestones, Awards, Certifications
    INTEREST = "interest"          # Hobbies, Passions, Pastimes
    CUSTOM = "custom"              # General persistent facts


class ImportanceLevel(int, Enum):
    """
    Numerical importance score from 1 (minor detail) to 10 (critical identity fact).
    """
    TRIVIAL = 1
    MINOR = 2
    MODERATE = 3
    STANDARD = 5
    IMPORTANT = 7
    HIGH = 8
    VERY_HIGH = 9
    CRITICAL = 10
