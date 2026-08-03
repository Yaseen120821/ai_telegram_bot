"""
app/rag/document_processing/loaders package initializer.
Provides LoaderFactory registry mapping DocumentType to appropriate format loader.
"""

import logging
from typing import Dict, Type

from app.rag.document_processing.loaders.base_loader import BaseDocumentLoader
from app.rag.document_processing.loaders.txt_loader import TXTLoader
from app.rag.document_processing.loaders.markdown_loader import MarkdownLoader
from app.rag.document_processing.loaders.code_loader import CodeLoader
from app.rag.document_processing.loaders.pdf_loader import PDFLoader
from app.rag.document_processing.loaders.docx_loader import DOCXLoader
from app.rag.document_processing.loaders.html_loader import HTMLLoader
from app.rag.document_processing.loaders.json_loader import JSONLoader
from app.rag.document_processing.loaders.csv_loader import CSVLoader
from app.rag.document_processing.document_types import DocumentType

logger = logging.getLogger("sana_ai.rag.loaders.factory")


class LoaderFactory:
    """
    Factory class instantiating appropriate BaseDocumentLoader based on DocumentType.
    """
    _LOADER_MAP: Dict[str, Type[BaseDocumentLoader]] = {
        DocumentType.TXT.value: TXTLoader,
        DocumentType.MARKDOWN.value: MarkdownLoader,
        DocumentType.CODE.value: CodeLoader,
        DocumentType.PYTHON.value: CodeLoader,
        DocumentType.JAVA.value: CodeLoader,
        DocumentType.CPP.value: CodeLoader,
        DocumentType.JAVASCRIPT.value: CodeLoader,
        DocumentType.TYPESCRIPT.value: CodeLoader,
        DocumentType.PDF.value: PDFLoader,
        DocumentType.DOCX.value: DOCXLoader,
        DocumentType.HTML.value: HTMLLoader,
        DocumentType.JSON.value: JSONLoader,
        DocumentType.CSV.value: CSVLoader,
        DocumentType.UNKNOWN.value: TXTLoader
    }

    @classmethod
    def get_loader(cls, file_type: str) -> BaseDocumentLoader:
        """
        Instantiates and returns the appropriate BaseDocumentLoader for a given file_type.

        Args:
            file_type (str): DocumentType value string.

        Returns:
            BaseDocumentLoader: Instantiated loader object.
        """
        key = file_type.lower() if file_type else DocumentType.UNKNOWN.value
        loader_cls = cls._LOADER_MAP.get(key, TXTLoader)
        logger.debug(f"⚙️ LoaderFactory instantiated '{loader_cls.__name__}' for file type '{key}'")
        return loader_cls()


__all__ = [
    "BaseDocumentLoader",
    "TXTLoader",
    "MarkdownLoader",
    "CodeLoader",
    "PDFLoader",
    "DOCXLoader",
    "HTMLLoader",
    "JSONLoader",
    "CSVLoader",
    "LoaderFactory"
]
