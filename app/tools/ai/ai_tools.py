"""
app/tools/ai/ai_tools.py - Local AI & Text Processing Analysis Tools
===================================================================
Provides deterministic local text processing tools: Summarize, Translate,
Grammar Check, Rewrite, Keyword Extraction, Text Statistics, and Sentiment Analysis.
"""

import re
import logging
from typing import Dict, Any, List, Optional
from app.tools.base_tool import BaseTool
from app.tools.tool_types import ToolCategory, PermissionLevel

logger = logging.getLogger("sana_ai.tools.ai")


class TextSummarizerTool(BaseTool):
    """Provides local extractive text summarization."""
    def __init__(self):
        super().__init__(
            name="summarize_text",
            description="Extracts key summary bullet points from text payload.",
            category=ToolCategory.AI,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "max_sentences": {"type": "integer"}
                },
                "required": ["text"]
            },
            tags=["summarize", "summary", "abstract", "key_points"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        text = parameters["text"]
        max_s = parameters.get("max_sentences", 3)

        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        summary = sentences[:max_s]
        return {
            "original_sentences": len(sentences),
            "summary_sentences": len(summary),
            "summary": " ".join(summary)
        }


class TextTranslatorTool(BaseTool):
    """Local simulation framework for text translation."""
    def __init__(self):
        super().__init__(
            name="translate_text",
            description="Translates text payload to a target language.",
            category=ToolCategory.AI,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "target_language": {"type": "string"}
                },
                "required": ["text", "target_language"]
            },
            tags=["translate", "translation", "language"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        text = parameters["text"]
        target = parameters["target_language"]
        return {
            "source_text": text,
            "target_language": target,
            "translated_text": f"[{target.upper()}] {text}"
        }


class GrammarCheckTool(BaseTool):
    """Checks basic spacing, punctuation, and capitalization issues."""
    def __init__(self):
        super().__init__(
            name="check_grammar",
            description="Inspects text for common spacing and capitalization errors.",
            category=ToolCategory.AI,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"}
                },
                "required": ["text"]
            },
            tags=["grammar", "proofread", "spellcheck"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        text = parameters["text"]
        corrections = []

        if re.search(r'\s{2,}', text):
            corrections.append("Multiple consecutive spaces detected and normalized.")

        corrected = re.sub(r'\s{2,}', ' ', text).strip()
        if corrected and not corrected[0].isupper():
            corrected = corrected[0].upper() + corrected[1:]
            corrections.append("Capitalized initial character.")

        return {
            "original_text": text,
            "corrected_text": corrected,
            "issues_found": len(corrections),
            "details": corrections
        }


class TextRewriterTool(BaseTool):
    """Rewrites input text into a specified tone style."""
    def __init__(self):
        super().__init__(
            name="rewrite_text",
            description="Rewrites input text to match professional, formal, or concise tone styles.",
            category=ToolCategory.AI,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "tone": {"type": "string", "enum": ["formal", "casual", "concise"]}
                },
                "required": ["text"]
            },
            tags=["rewrite", "paraphrase", "tone", "style"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        text = parameters["text"]
        tone = parameters.get("tone", "formal").lower()

        if tone == "concise":
            rewritten = text.split(".")[0] + "." if "." in text else text
        elif tone == "formal":
            rewritten = f"It is noted that: {text}"
        else:
            rewritten = f"Hey, basically: {text}"

        return {"tone": tone, "original": text, "rewritten": rewritten}


class KeywordExtractorTool(BaseTool):
    """Extracts top frequency keywords from text."""
    def __init__(self):
        super().__init__(
            name="extract_keywords",
            description="Extracts high-frequency keywords and topic terms from text payload.",
            category=ToolCategory.AI,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "top_n": {"type": "integer"}
                },
                "required": ["text"]
            },
            tags=["keywords", "tags", "topics", "n_grams"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        text = parameters["text"]
        top_n = min(max(parameters.get("top_n", 5), 1), 20)

        stopwords = {"the", "a", "an", "is", "are", "was", "were", "and", "or", "in", "on", "at", "to", "for", "of", "with", "this", "that"}
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        filtered = [w for w in words if w not in stopwords]

        freq: Dict[str, int] = {}
        for w in filtered:
            freq[w] = freq.get(w, 0) + 1

        sorted_kw = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return {
            "top_n": top_n,
            "keywords": [k for k, v in sorted_kw],
            "frequencies": dict(sorted_kw)
        }


class TextStatisticsTool(BaseTool):
    """Computes word count, character count, and reading time."""
    def __init__(self):
        super().__init__(
            name="text_statistics",
            description="Calculates total words, characters, sentences, and estimated reading time.",
            category=ToolCategory.AI,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"}
                },
                "required": ["text"]
            },
            tags=["stats", "word_count", "reading_time"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        text = parameters["text"]
        words = text.split()
        chars = len(text)
        sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
        read_time_min = round(len(words) / 200.0, 2)

        return {
            "word_count": len(words),
            "character_count": chars,
            "sentence_count": len(sentences),
            "estimated_reading_time_minutes": read_time_min
        }


class SentimentAnalysisTool(BaseTool):
    """Analyzes text sentiment (positive, neutral, negative)."""
    def __init__(self):
        super().__init__(
            name="analyze_sentiment",
            description="Evaluates positive, neutral, or negative sentiment score for text.",
            category=ToolCategory.AI,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"}
                },
                "required": ["text"]
            },
            tags=["sentiment", "polarity", "emotion"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        text = parameters["text"].lower()

        pos_words = {"good", "great", "excellent", "happy", "awesome", "fantastic", "positive", "love", "wonderful", "success"}
        neg_words = {"bad", "terrible", "horrible", "sad", "awful", "negative", "fail", "failure", "hate", "error", "broken"}

        pos_score = sum(1 for w in pos_words if w in text)
        neg_score = sum(1 for w in neg_words if w in text)

        if pos_score > neg_score:
            sentiment = "positive"
            score = 0.8
        elif neg_score > pos_score:
            sentiment = "negative"
            score = -0.8
        else:
            sentiment = "neutral"
            score = 0.0

        return {
            "sentiment": sentiment,
            "confidence_score": score,
            "positive_triggers": pos_score,
            "negative_triggers": neg_score
        }
