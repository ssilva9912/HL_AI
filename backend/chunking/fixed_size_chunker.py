from backend.interfaces.chunker import DocumentChunk
from backend.interfaces.parser import ParsedDocument


class FixedSizeChunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap cannot be negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: ParsedDocument) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        content = document.content

        if not content:
            return chunks

        start = 0
        chunk_index = 0

        while start < len(content):
            end = min(start + self.chunk_size, len(content))

            chunks.append(
                DocumentChunk(
                    source_document=document,
                    content=content[start:end],
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end,
                )
            )

            if end == len(content):
                break

            start = end - self.overlap
            chunk_index += 1

        return chunks