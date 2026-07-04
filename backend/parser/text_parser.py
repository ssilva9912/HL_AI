from backend.ingestion.scanner import FileMetadata
from backend.interfaces.parser import ParsedDocument


class TextParser:
    def can_parse(self, file: FileMetadata) -> bool:
        return file.path.suffix.lower() == ".txt"

    def parse(self, file: FileMetadata) -> ParsedDocument:
        content = file.path.read_text(encoding="utf-8")

        return ParsedDocument(
            source_path=file.path,
            file_type="text",
            content=content,
            metadata=file,
        )