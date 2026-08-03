"""
app/rag/document_processing package initializer.
Exposes public API for Document Parsing Loaders, Validator, Cleaner, Chunker, Metadata Extractor, and DocumentManager.
"""

from app.rag.document_processing.document_config import (
    MAX_FILE_SIZE_BYTES,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    MIN_CHUNK_SIZE,
    MAX_CHUNK_SIZE,
    DEFAULT_ENCODING,
    FALLBACK_ENCODINGS,
    KNOWLEDGE_BASE_DIR,
    PROCESSED_CHUNKS_DIR
)
from app.rag.document_processing.document_types import (
    DocumentType,
    ProcessingStatus,
    ChunkingStrategy
)
from app.rag.document_processing.document_models import (
    Document,
    DocumentMetadata,
    Chunk,
    ChunkCollection,
    DocumentStatistics,
    ProcessingResult,
    ValidationResult,
    CleaningResult
)
from app.rag.document_processing.document_utils import DocumentUtils
from app.rag.document_processing.loaders import (
    BaseDocumentLoader,
    TXTLoader,
    MarkdownLoader,
    CodeLoader,
    PDFLoader,
    DOCXLoader,
    HTMLLoader,
    JSONLoader,
    CSVLoader,
    LoaderFactory
)
from app.rag.document_processing.document_validator import DocumentValidator
from app.rag.document_processing.document_cleaner import DocumentCleaner
from app.rag.document_processing.document_metadata import MetadataExtractor
from app.rag.document_processing.document_chunker import DocumentChunker
from app.rag.document_processing.document_manager import DocumentManager

__all__ = [
    "MAX_FILE_SIZE_BYTES",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
    "MIN_CHUNK_SIZE",
    "MAX_CHUNK_SIZE",
    "DEFAULT_ENCODING",
    "FALLBACK_ENCODINGS",
    "KNOWLEDGE_BASE_DIR",
    "PROCESSED_CHUNKS_DIR",
    "DocumentType",
    "ProcessingStatus",
    "ChunkingStrategy",
    "Document",
    "DocumentMetadata",
    "Chunk",
    "ChunkCollection",
    "DocumentStatistics",
    "ProcessingResult",
    "ValidationResult",
    "CleaningResult",
    "DocumentUtils",
    "BaseDocumentLoader",
    "TXTLoader",
    "MarkdownLoader",
    "CodeLoader",
    "PDFLoader",
    "DOCXLoader",
    "HTMLLoader",
    "JSONLoader",
    "CSVLoader",
    "LoaderFactory",
    "DocumentValidator",
    "DocumentCleaner",
    "MetadataExtractor",
    "DocumentChunker",
    "DocumentManager"
]
