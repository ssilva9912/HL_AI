from pathlib import Path

from backend.chunking.semantic_chunker import SemanticChunker
from backend.indexing.models import ProcessedDocument
from backend.ingestion.scanner import FileMetadata
from backend.interfaces.chunker import Chunker
from backend.interfaces.parser import Parser
from backend.parser.markdown_parser import MarkdownParser
from backend.parser.pdf_parser import PdfParser
from backend.parser.registry import ParserRegistry
from backend.parser.text_parser import TextParser


class DocumentProcessor:
    def __init__(
        self,
        parser: Parser | None = None,
        chunker: Chunker | None = None,
    ) -> None:
        self._parser = parser or ParserRegistry(
            parsers=[
                TextParser(),
                MarkdownParser(),
                PdfParser(),
            ]
        )

        self._chunker = chunker or SemanticChunker()

    def process_file(
        self,
        path: Path,
    ) -> ProcessedDocument:
        metadata = self._build_metadata(path)

        if not self._parser.can_parse(metadata):
            raise ValueError(f"Unsupported file type: {path.suffix}")

        document = self._parser.parse(metadata)
        chunks = self._chunker.chunk(document)

        return ProcessedDocument(
            document=document,
            chunks=chunks,
        )

    def process_paths(
        self,
        paths: list[Path],
    ) -> list[ProcessedDocument]:
        processed: list[ProcessedDocument] = []

        for path in paths:
            try:
                processed.append(self.process_file(path))
            except ValueError as exc:
                if str(exc).startswith("Unsupported file type:"):
                    continue
                raise

        return processed

    @staticmethod
    def _build_metadata(path: Path) -> FileMetadata:
        if not path.exists():
            raise FileNotFoundError(f"file does not exist: {path}")

        if not path.is_file():
            raise ValueError(f"path is not a file: {path}")

        return FileMetadata(
            name=path.name,
            path=path,
            extension=path.suffix.lower(),
            size_bytes=path.stat().st_size,
        )
