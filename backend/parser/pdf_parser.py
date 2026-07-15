from pypdf import PdfReader

from backend.ingestion.scanner import FileMetadata
from backend.interfaces.parser import ParsedDocument


class PdfParser:
    def can_parse(self, file: FileMetadata) -> bool:
        return file.path.suffix.lower() == ".pdf"

    def parse(self, file: FileMetadata) -> ParsedDocument:
        reader = PdfReader(file.path)

        page_texts = [
            extracted_text.strip()
            for page in reader.pages
            if (extracted_text := page.extract_text()) and extracted_text.strip()
        ]

        content = "\n\n".join(page_texts)

        if not content:
            raise ValueError(
                f"PDF contains no extractable text: {file.name}. "
                "The document may be scanned and require OCR."
            )

        return ParsedDocument(
            source_path=file.path,
            file_type="pdf",
            content=content,
            metadata=file,
        )
