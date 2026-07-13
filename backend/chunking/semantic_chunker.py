import re
from dataclasses import dataclass

from backend.interfaces.chunker import DocumentChunk
from backend.interfaces.parser import ParsedDocument


@dataclass(frozen=True)
class TextSegment:
    content: str
    start_char: int
    end_char: int


class SemanticChunker:
    """Split documents into sentence-aware chunks.

    The chunker prefers sentence and paragraph boundaries while enforcing a
    maximum character length. Long sentences are split on whitespace as a
    fallback.
    """

    SENTENCE_PATTERN = re.compile(r".+?(?:[.!?](?=\s|$)|$)", re.DOTALL)

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
        if not document.content or not document.content.strip():
            return []

        segments = self._split_into_segments(document.content)

        if not segments:
            return []

        chunk_ranges = self._group_segments(segments)

        return [
            DocumentChunk(
                source_document=document,
                content=document.content[start:end],
                chunk_index=index,
                start_char=start,
                end_char=end,
            )
            for index, (start, end) in enumerate(chunk_ranges)
        ]

    def _split_into_segments(self, content: str) -> list[TextSegment]:
        segments: list[TextSegment] = []

        for match in self.SENTENCE_PATTERN.finditer(content):
            raw_start = match.start()
            raw_end = match.end()
            raw_content = match.group()

            leading_whitespace = len(raw_content) - len(raw_content.lstrip())
            trailing_whitespace = len(raw_content) - len(raw_content.rstrip())

            start = raw_start + leading_whitespace
            end = raw_end - trailing_whitespace

            if start >= end:
                continue

            segments.extend(self._split_long_segment(content, start, end))

        return segments

    def _split_long_segment(
        self,
        content: str,
        start: int,
        end: int,
    ) -> list[TextSegment]:
        if end - start <= self.chunk_size:
            return [
                TextSegment(
                    content=content[start:end],
                    start_char=start,
                    end_char=end,
                )
            ]

        segments: list[TextSegment] = []
        current_start = start

        while current_start < end:
            maximum_end = min(current_start + self.chunk_size, end)

            if maximum_end == end:
                split_end = end
            else:
                whitespace_position = content.rfind(
                    " ",
                    current_start,
                    maximum_end + 1,
                )

                if whitespace_position > current_start:
                    split_end = whitespace_position
                else:
                    split_end = maximum_end

            trimmed_end = split_end

            while trimmed_end > current_start and content[trimmed_end - 1].isspace():
                trimmed_end -= 1

            if trimmed_end > current_start:
                segments.append(
                    TextSegment(
                        content=content[current_start:trimmed_end],
                        start_char=current_start,
                        end_char=trimmed_end,
                    )
                )

            current_start = split_end

            while current_start < end and content[current_start].isspace():
                current_start += 1

        return segments

    def _group_segments(
        self,
        segments: list[TextSegment],
    ) -> list[tuple[int, int]]:
        chunk_ranges: list[tuple[int, int]] = []
        current_segments: list[TextSegment] = []

        for segment in segments:
            if not current_segments:
                current_segments.append(segment)
                continue

            candidate_size = segment.end_char - current_segments[0].start_char

            if candidate_size <= self.chunk_size:
                current_segments.append(segment)
                continue

            chunk_ranges.append(
                (
                    current_segments[0].start_char,
                    current_segments[-1].end_char,
                )
            )

            current_segments = self._overlap_segments(current_segments)

            while (
                current_segments
                and segment.end_char - current_segments[0].start_char > self.chunk_size
            ):
                current_segments.pop(0)

            current_segments.append(segment)

        if current_segments:
            chunk_ranges.append(
                (
                    current_segments[0].start_char,
                    current_segments[-1].end_char,
                )
            )

        return chunk_ranges

    def _overlap_segments(
        self,
        segments: list[TextSegment],
    ) -> list[TextSegment]:
        if self.overlap == 0:
            return []

        overlap_segments: list[TextSegment] = []
        overlap_size = 0

        for segment in reversed(segments):
            segment_size = segment.end_char - segment.start_char

            if overlap_segments and overlap_size + segment_size > self.overlap:
                break

            overlap_segments.insert(0, segment)
            overlap_size = segments[-1].end_char - segment.start_char

            if overlap_size >= self.overlap:
                break

        return overlap_segments
