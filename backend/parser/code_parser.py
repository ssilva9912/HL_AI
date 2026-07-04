from backend.ingestion.scanner import FileMetadata
from backend.interfaces.parser import ParsedDocument


class CodeParser:
    def can_parse(self, file: FileMetadata) -> bool:
        return file.path.suffix.lower() == ".py"

    def parse(self, file: FileMetadata) -> ParsedDocument:
        content = file.path.read_text(encoding="utf-8")

        return ParsedDocument(
            source_path=file.path,
            file_type="python",
            content=content,
            metadata=file,
        )