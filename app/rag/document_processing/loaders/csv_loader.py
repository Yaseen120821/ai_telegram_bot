"""
app/rag/document_processing/loaders/csv_loader.py - CSV Document Loader
========================================================================
"""

import csv
import logging
from app.rag.document_processing.loaders.base_loader import BaseDocumentLoader

logger = logging.getLogger("sana_ai.rag.loaders.csv")


class CSVLoader(BaseDocumentLoader):
    """
    Parser for CSV (.csv) files, formatting rows into readable text statements.
    """

    def load(self, filepath: str) -> str:
        logger.debug(f"📈 Loading CSV Document: {filepath}")
        raw = self.read_with_fallback_encoding(filepath)
        try:
            lines = []
            reader = csv.reader(raw.splitlines())
            header = None
            for idx, row in enumerate(reader):
                if idx == 0:
                    header = row
                    lines.append(f"Columns: {', '.join(row)}")
                else:
                    if header and len(row) == len(header):
                        row_str = ", ".join([f"{h}: {v}" for h, v in zip(header, row)])
                        lines.append(f"Row {idx}: {row_str}")
                    else:
                        lines.append(f"Row {idx}: {', '.join(row)}")
            return "\n".join(lines)
        except Exception:
            return raw
