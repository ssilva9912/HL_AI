from collections import defaultdict
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from qdrant_client import QdrantClient, models

from backend.ingestion.scanner import FileMetadata
from backend.interfaces.chunker import DocumentChunk
from backend.interfaces.embedder import EmbeddedChunk
from backend.interfaces.parser import ParsedDocument
from backend.interfaces.vector_store import SearchResult


class QdrantVectorStore:
    def __init__(
        self,
        storage_path: Path,
        collection_name: str = "homelab_documents",
    ) -> None:
        if not collection_name.strip():
            raise ValueError("collection_name must not be empty")

        storage_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._collection_name = collection_name
        self._client = QdrantClient(
            path=str(storage_path),
        )

    def add(
        self,
        embedded_chunk: EmbeddedChunk,
    ) -> None:
        document_version = self._document_version([embedded_chunk])
        points = self._build_points(
            [embedded_chunk],
            document_version,
        )

        self._upsert(
            points,
            vector_size=len(embedded_chunk.vector),
        )

    def add_many(
        self,
        embedded_chunks: list[EmbeddedChunk],
    ) -> None:
        grouped_chunks: dict[
            str,
            list[EmbeddedChunk],
        ] = defaultdict(list)

        for embedded_chunk in embedded_chunks:
            document_name = embedded_chunk.chunk.source_document.metadata.name
            grouped_chunks[document_name].append(embedded_chunk)

        for document_name, document_chunks in grouped_chunks.items():
            self.replace_document(
                document_name,
                document_chunks,
            )

    def replace_document(
        self,
        document_name: str,
        embedded_chunks: Sequence[EmbeddedChunk],
        *,
        document_id: UUID | None = None,
    ) -> None:
        if not document_name.strip():
            raise ValueError("document_name must not be empty")

        if not embedded_chunks:
            raise ValueError("embedded_chunks must not be empty")

        chunk_document_names = {
            embedded_chunk.chunk.source_document.metadata.name for embedded_chunk in embedded_chunks
        }

        if chunk_document_names != {document_name}:
            raise ValueError("all chunks must belong to document_name")

        document_version = self._document_version(embedded_chunks)
        points = self._build_points(
            embedded_chunks,
            document_version,
            document_id=document_id,
        )

        self._upsert(
            points,
            vector_size=len(embedded_chunks[0].vector),
        )

        scope_key, scope_value = self._document_scope(
            document_name,
            document_id,
        )

        stale_points = models.Filter(
            must=[
                models.FieldCondition(
                    key=scope_key,
                    match=models.MatchValue(
                        value=scope_value,
                    ),
                )
            ],
            must_not=[
                models.FieldCondition(
                    key="document_version",
                    match=models.MatchValue(
                        value=document_version,
                    ),
                )
            ],
        )

        self._client.delete(
            collection_name=self._collection_name,
            points_selector=models.FilterSelector(
                filter=stale_points,
            ),
            wait=True,
        )

        if document_id is not None:
            self._delete_legacy_document_points(
                document_name,
            )

    def delete_document(
        self,
        document_name: str,
        *,
        document_id: UUID | None = None,
    ) -> None:
        if not document_name.strip():
            raise ValueError("document_name must not be empty")

        if not self._collection_exists():
            return

        scope_key, scope_value = self._document_scope(
            document_name,
            document_id,
        )

        self._client.delete(
            collection_name=self._collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key=scope_key,
                            match=models.MatchValue(
                                value=scope_value,
                            ),
                        )
                    ]
                )
            ),
            wait=True,
        )

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if not query_vector:
            raise ValueError("query_vector must not be empty")

        if not self._collection_exists():
            return []

        response = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=True,
        )

        document_cache: dict[
            str,
            ParsedDocument,
        ] = {}

        return [
            SearchResult(
                embedded_chunk=self._record_to_embedded_chunk(
                    point,
                    document_cache,
                ),
                score=float(point.score),
            )
            for point in response.points
        ]

    def count(self) -> int:
        if not self._collection_exists():
            return 0

        result = self._client.count(
            collection_name=self._collection_name,
            exact=True,
        )

        return int(result.count)

    def items(self) -> list[EmbeddedChunk]:
        return self._scroll_items()

    def document_items(
        self,
        document_name: str,
        *,
        document_id: UUID | None = None,
    ) -> list[EmbeddedChunk]:
        if not document_name.strip():
            raise ValueError("document_name must not be empty")

        scope_key, scope_value = self._document_scope(
            document_name,
            document_id,
        )

        return self._scroll_items(
            models.Filter(
                must=[
                    models.FieldCondition(
                        key=scope_key,
                        match=models.MatchValue(
                            value=scope_value,
                        ),
                    )
                ],
            )
        )

    def _scroll_items(
        self,
        scroll_filter: models.Filter | None = None,
    ) -> list[EmbeddedChunk]:
        if not self._collection_exists():
            return []

        records: list[Any] = []
        offset: Any = None

        while True:
            page, offset = self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=scroll_filter,
                offset=offset,
                limit=256,
                with_payload=True,
                with_vectors=True,
            )

            records.extend(page)

            if offset is None:
                break

        document_cache: dict[
            str,
            ParsedDocument,
        ] = {}

        embedded_chunks = [
            self._record_to_embedded_chunk(
                record,
                document_cache,
            )
            for record in records
        ]

        return sorted(
            embedded_chunks,
            key=lambda item: (
                item.chunk.source_document.metadata.name,
                item.chunk.chunk_index,
                item.chunk.start_char,
            ),
        )

    def clear(self) -> None:
        if self._collection_exists():
            self._client.delete_collection(self._collection_name)

    def close(self) -> None:
        self._client.close()

    def _collection_exists(self) -> bool:
        return self._client.collection_exists(self._collection_name)

    def _ensure_collection(
        self,
        vector_size: int,
    ) -> None:
        if vector_size <= 0:
            raise ValueError("vectors must not be empty")

        if self._collection_exists():
            return

        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    def _upsert(
        self,
        points: Sequence[models.PointStruct],
        vector_size: int,
    ) -> None:
        self._ensure_collection(vector_size)

        self._client.upsert(
            collection_name=self._collection_name,
            points=points,
            wait=True,
        )

    def _delete_legacy_document_points(
        self,
        document_name: str,
    ) -> None:
        legacy_points = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_name",
                    match=models.MatchValue(
                        value=document_name,
                    ),
                ),
                models.IsEmptyCondition(
                    is_empty=models.PayloadField(
                        key="document_id",
                    ),
                ),
            ],
        )

        self._client.delete(
            collection_name=self._collection_name,
            points_selector=models.FilterSelector(
                filter=legacy_points,
            ),
            wait=True,
        )

    @staticmethod
    def _document_scope(
        document_name: str,
        document_id: UUID | None,
    ) -> tuple[str, str]:
        if document_id is None:
            return "document_name", document_name

        return "document_id", str(document_id)

    @classmethod
    def _build_points(
        cls,
        embedded_chunks: Sequence[EmbeddedChunk],
        document_version: str,
        *,
        document_id: UUID | None = None,
    ) -> list[models.PointStruct]:
        vector_sizes = {len(embedded_chunk.vector) for embedded_chunk in embedded_chunks}

        if len(vector_sizes) != 1 or 0 in vector_sizes:
            raise ValueError(
                "all vectors must have the same non-zero dimension",
            )

        points: list[models.PointStruct] = []

        for embedded_chunk in embedded_chunks:
            chunk = embedded_chunk.chunk
            document = chunk.source_document
            metadata = document.metadata

            document_identity = str(document_id) if document_id is not None else metadata.name

            point_key = (
                f"{document_identity}:{document_version}:"
                f"{chunk.chunk_index}:"
                f"{chunk.start_char}:"
                f"{chunk.end_char}"
            )

            payload: dict[str, Any] = {
                "document_name": metadata.name,
                "document_version": document_version,
                "source_path": str(document.source_path),
                "extension": metadata.extension,
                "size_bytes": metadata.size_bytes,
                "file_type": document.file_type,
                "chunk_content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
            }

            if document_id is not None:
                payload["document_id"] = str(document_id)

            points.append(
                models.PointStruct(
                    id=str(
                        uuid5(
                            NAMESPACE_URL,
                            point_key,
                        )
                    ),
                    vector=embedded_chunk.vector,
                    payload=payload,
                )
            )

        return points

    @staticmethod
    def _document_version(
        embedded_chunks: Sequence[EmbeddedChunk],
    ) -> str:
        hasher = sha256()

        for embedded_chunk in sorted(
            embedded_chunks,
            key=lambda item: (
                item.chunk.chunk_index,
                item.chunk.start_char,
                item.chunk.end_char,
            ),
        ):
            chunk = embedded_chunk.chunk

            hasher.update(str(chunk.chunk_index).encode())
            hasher.update(str(chunk.start_char).encode())
            hasher.update(str(chunk.end_char).encode())
            hasher.update(chunk.content.encode("utf-8"))

        return hasher.hexdigest()

    @classmethod
    def _record_to_embedded_chunk(
        cls,
        record: Any,
        document_cache: dict[
            str,
            ParsedDocument,
        ],
    ) -> EmbeddedChunk:
        payload = dict(record.payload or {})

        document_name = cls._required_str(
            payload,
            "document_name",
        )
        source_path = Path(
            cls._required_str(
                payload,
                "source_path",
            )
        )
        document_key = f"{source_path}:{document_name}"

        document = document_cache.get(document_key)

        if document is None:
            metadata = FileMetadata(
                name=document_name,
                path=source_path,
                extension=cls._required_str(
                    payload,
                    "extension",
                ),
                size_bytes=cls._required_int(
                    payload,
                    "size_bytes",
                ),
            )

            document = ParsedDocument(
                source_path=source_path,
                file_type=cls._required_str(
                    payload,
                    "file_type",
                ),
                content="",
                metadata=metadata,
            )

            document_cache[document_key] = document

        raw_vector = record.vector

        if not isinstance(raw_vector, list):
            raise ValueError(
                "Qdrant point does not contain a dense vector",
            )

        chunk = DocumentChunk(
            source_document=document,
            content=cls._required_str(
                payload,
                "chunk_content",
            ),
            chunk_index=cls._required_int(
                payload,
                "chunk_index",
            ),
            start_char=cls._required_int(
                payload,
                "start_char",
            ),
            end_char=cls._required_int(
                payload,
                "end_char",
            ),
        )

        return EmbeddedChunk(
            chunk=chunk,
            vector=[float(value) for value in raw_vector],
        )

    @staticmethod
    def _required_str(
        payload: Mapping[str, Any],
        key: str,
    ) -> str:
        value = payload.get(key)

        if not isinstance(value, str):
            raise ValueError(
                f"Qdrant payload field {key!r} must be a string",
            )

        return value

    @staticmethod
    def _required_int(
        payload: Mapping[str, Any],
        key: str,
    ) -> int:
        value = payload.get(key)

        if not isinstance(value, int):
            raise ValueError(
                f"Qdrant payload field {key!r} must be an integer",
            )

        return value
