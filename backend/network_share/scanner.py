import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from uuid import UUID

from sqlalchemy.orm import Session

from backend.database import (
    NetworkShareFileRepository,
    NetworkShareSource,
    NetworkShareSourceRepository,
    NetworkShareStatus,
)

SessionFactory = Callable[[], Session]
DEFAULT_DOCUMENT_EXTENSIONS = frozenset({".md", ".pdf", ".txt"})


@dataclass(frozen=True)
class ScannedShareFile:
    relative_path: str
    checksum_sha256: str
    size_bytes: int
    modified_time_ns: int


@dataclass(frozen=True)
class NetworkShareScanResult:
    source_id: UUID
    discovered: int
    changed: int
    unchanged: int
    missing: int
    unsupported: int
    unsafe: int
    errors: tuple[str, ...] = ()


class NetworkShareUnavailableError(RuntimeError):
    """Raised when a configured share cannot be scanned completely."""


class NetworkShareScanner:
    """Observe a read-only directory and persist its inventory.

    Scanning deliberately does not copy, queue, index, or delete documents.
    A file is marked missing only after a complete, successful directory walk.
    """

    def __init__(
        self,
        *,
        session_factory: SessionFactory,
        supported_extensions: Iterable[str] = DEFAULT_DOCUMENT_EXTENSIONS,
    ) -> None:
        self._session_factory = session_factory
        self._supported_extensions = frozenset(
            extension.casefold() for extension in supported_extensions
        )

    def scan(self, *, name: str, root_path: Path) -> NetworkShareScanResult:
        started_at = datetime.now(UTC)
        source_id = self._begin_scan(name=name, root_path=root_path, started_at=started_at)

        try:
            scanned, unsupported, unsafe = self._walk(root_path)
        except (OSError, NetworkShareUnavailableError) as exc:
            self._finish_failed_scan(source_id=source_id, error=str(exc))
            raise NetworkShareUnavailableError(str(exc)) from exc

        seen_at = datetime.now(UTC)
        discovered = 0
        changed = 0
        unchanged = 0

        with self._session_factory() as session, session.begin():
            source = self._require_source(session, source_id)
            files = NetworkShareFileRepository(session)
            seen_paths: set[str] = set()
            for scanned_file in scanned:
                _, is_new, was_changed = files.record_seen(
                    source=source,
                    relative_path=scanned_file.relative_path,
                    checksum_sha256=scanned_file.checksum_sha256,
                    size_bytes=scanned_file.size_bytes,
                    modified_time_ns=scanned_file.modified_time_ns,
                    seen_at=seen_at,
                )
                seen_paths.add(scanned_file.relative_path)
                if is_new:
                    discovered += 1
                elif was_changed:
                    changed += 1
                else:
                    unchanged += 1

            missing = files.mark_unseen_missing(
                source_id=source_id,
                seen_paths=seen_paths,
            )
            NetworkShareSourceRepository(session).update_scan_status(
                source,
                status=NetworkShareStatus.READY,
                completed_at=seen_at,
            )

        return NetworkShareScanResult(
            source_id=source_id,
            discovered=discovered,
            changed=changed,
            unchanged=unchanged,
            missing=missing,
            unsupported=unsupported,
            unsafe=unsafe,
        )

    def _begin_scan(
        self,
        *,
        name: str,
        root_path: Path,
        started_at: datetime,
    ) -> UUID:
        with self._session_factory() as session, session.begin():
            repositories = NetworkShareSourceRepository(session)
            source = repositories.get_or_create(name=name, root_path=str(root_path))
            repositories.update_scan_status(
                source,
                status=NetworkShareStatus.SYNCING,
                started_at=started_at,
                error=None,
            )
            return source.id

    def _finish_failed_scan(self, *, source_id: UUID, error: str) -> None:
        with self._session_factory() as session, session.begin():
            source = self._require_source(session, source_id)
            NetworkShareSourceRepository(session).update_scan_status(
                source,
                status=NetworkShareStatus.UNAVAILABLE,
                completed_at=datetime.now(UTC),
                error=error,
            )

    def _walk(self, root_path: Path) -> tuple[list[ScannedShareFile], int, int]:
        if not root_path.exists():
            raise NetworkShareUnavailableError(f"Share path does not exist: {root_path}")
        if not root_path.is_dir():
            raise NetworkShareUnavailableError(f"Share path is not a directory: {root_path}")

        scanned: list[ScannedShareFile] = []
        unsupported = 0
        unsafe = 0
        walk_errors: list[str] = []

        def record_walk_error(error: OSError) -> None:
            walk_errors.append(str(error))

        for directory, directory_names, filenames in os.walk(
            root_path,
            followlinks=False,
            onerror=record_walk_error,
        ):
            directory_path = Path(directory)
            safe_directories: list[str] = []
            for directory_name in directory_names:
                candidate = directory_path / directory_name
                if candidate.is_symlink():
                    unsafe += 1
                else:
                    safe_directories.append(directory_name)
            directory_names[:] = safe_directories

            for filename in filenames:
                candidate = directory_path / filename
                if candidate.is_symlink():
                    unsafe += 1
                    continue
                if candidate.suffix.casefold() not in self._supported_extensions:
                    unsupported += 1
                    continue
                scanned.append(self._inspect_file(root_path, candidate))

        if walk_errors:
            raise NetworkShareUnavailableError("; ".join(walk_errors))
        return scanned, unsupported, unsafe

    @staticmethod
    def _inspect_file(root_path: Path, path: Path) -> ScannedShareFile:
        before = path.stat()
        digest = sha256()
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise NetworkShareUnavailableError(f"File changed while being scanned: {path}")
        return ScannedShareFile(
            relative_path=path.relative_to(root_path).as_posix(),
            checksum_sha256=digest.hexdigest(),
            size_bytes=after.st_size,
            modified_time_ns=after.st_mtime_ns,
        )

    @staticmethod
    def _require_source(session: Session, source_id: UUID) -> NetworkShareSource:
        source = session.get(NetworkShareSource, source_id)
        if source is None:
            raise RuntimeError(f"Network-share source disappeared during scan: {source_id}")
        return source
