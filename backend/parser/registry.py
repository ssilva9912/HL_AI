from collections.abc import Sequence

from backend.ingestion.scanner import FileMetadata
from backend.interfaces.parser import ParsedDocument, Parser


class ParserRegistry:
    """
    Select the appropriate parser for a document based on parser support.

    Parsers are checked in registration order. The first parser whose
    can_parse method returns True is used.
    """

    def __init__(self, parsers: Sequence[Parser]) -> None:
        if not parsers:
            raise ValueError("at least one parser is required")

        self._parsers = list(parsers)

    def can_parse(self, file: FileMetadata) -> bool:
        return any(parser.can_parse(file) for parser in self._parsers)

    def parse(self, file: FileMetadata) -> ParsedDocument:
        parser = self.get_parser(file)
        return parser.parse(file)

    def get_parser(self, file: FileMetadata) -> Parser:
        for parser in self._parsers:
            if parser.can_parse(file):
                return parser

        raise ValueError(f"no parser is registered for extension: {file.extension}")
