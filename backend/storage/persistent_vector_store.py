import os
import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from backend.interfaces.embedder import EmbeddedChunk
from backend.storage.in_memory_vector_store import InMemoryVectorStore


class PersistentVectorStore(InMemoryVectorStore):
    """
    File-backed vector store for trusted local Homelab AI data.

    The persistence file uses Python pickle so complete EmbeddedChunk object
    graphs can be restored without duplicating model serialization logic.

    Pickle files must never be loaded from untrusted sources.
    """

    def __init__(
        self,
        storage_path: Path,
        load_existing: bool = True,
    ) -> None:
        self._storage_path = storage_path

        embedded_chunks: list[EmbeddedChunk] = []

        if load_existing:
            embedded_chunks = self._load()

        super().__init__(embedded_chunks)

    @property
    def storage_path(self) -> Path:
        return self._storage_path

    def add(self, embedded_chunk: EmbeddedChunk) -> None:
        super().add(embedded_chunk)
        self.save()

    def add_many(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        if not embedded_chunks:
            return

        super().add_many(embedded_chunks)
        self.save()

    def clear(self) -> None:
        super().clear()
        self.save()

    def replace_all(
        self,
        embedded_chunks: Sequence[EmbeddedChunk],
    ) -> None:
        self._items = list(embedded_chunks)
        self.save()

    def save(self) -> None:
        self._storage_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = self._storage_path.with_suffix(f"{self._storage_path.suffix}.tmp")

        try:
            with temporary_path.open("wb") as file:
                pickle.dump(
                    self._items,
                    file,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

                file.flush()
                os.fsync(file.fileno())

            temporary_path.replace(self._storage_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    def _load(self) -> list[EmbeddedChunk]:
        if not self._storage_path.exists():
            return []

        if not self._storage_path.is_file():
            raise ValueError(f"vector-store path is not a file: {self._storage_path}")

        try:
            with self._storage_path.open("rb") as file:
                stored_value: Any = pickle.load(file)
        except (
            EOFError,
            OSError,
            pickle.PickleError,
        ) as exc:
            raise ValueError(f"could not load vector store: {self._storage_path}") from exc

        if not isinstance(stored_value, list):
            raise ValueError(f"invalid vector-store data: {self._storage_path}")

        if not all(isinstance(item, EmbeddedChunk) for item in stored_value):
            raise ValueError(f"vector store contains invalid entries: {self._storage_path}")

        return stored_value
