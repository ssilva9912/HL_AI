from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from backend.database import (
    Base,
    NetworkShareFile,
    NetworkShareFileStatus,
    NetworkShareSource,
    NetworkShareStatus,
)
from backend.network_share import NetworkShareScanner
from backend.network_share.scanner import NetworkShareUnavailableError


@pytest.fixture
def session_factory(tmp_path: Path) -> Iterator[sessionmaker[Session]]:
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'scanner.db'}")
    factory = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)
    Base.metadata.create_all(engine)
    try:
        yield factory
    finally:
        engine.dispose()


def test_scan_tracks_supported_files_and_ignores_unsupported(
    tmp_path: Path,
    session_factory: sessionmaker[Session],
) -> None:
    share = tmp_path / "share"
    nested = share / "team"
    nested.mkdir(parents=True)
    (share / "notes.txt").write_text("hello", encoding="utf-8")
    (nested / "guide.md").write_text("guide", encoding="utf-8")
    (share / "photo.jpg").write_bytes(b"not a document")

    result = NetworkShareScanner(session_factory=session_factory).scan(
        name="Local test share",
        root_path=share,
    )

    assert result.discovered == 2
    assert result.changed == 0
    assert result.unchanged == 0
    assert result.missing == 0
    assert result.unsupported == 1
    with session_factory() as session:
        source = session.scalar(select(NetworkShareSource))
        assert source is not None
        assert source.status is NetworkShareStatus.READY
        tracked = list(
            session.scalars(select(NetworkShareFile).order_by(NetworkShareFile.relative_path))
        )
        assert [item.relative_path for item in tracked] == ["notes.txt", "team/guide.md"]
        assert all(item.status is NetworkShareFileStatus.DISCOVERED for item in tracked)


def test_rescan_classifies_unchanged_changed_missing_and_recovered_files(
    tmp_path: Path,
    session_factory: sessionmaker[Session],
) -> None:
    share = tmp_path / "share"
    share.mkdir()
    stable = share / "stable.txt"
    changing = share / "changing.md"
    recovered = share / "recovered.pdf"
    stable.write_text("stable", encoding="utf-8")
    changing.write_text("before", encoding="utf-8")
    recovered.write_bytes(b"pdf")
    scanner = NetworkShareScanner(session_factory=session_factory)
    scanner.scan(name="Local test share", root_path=share)

    changing.write_text("after", encoding="utf-8")
    recovered.unlink()
    second = scanner.scan(name="Local test share", root_path=share)
    assert second.discovered == 0
    assert second.changed == 1
    assert second.unchanged == 1
    assert second.missing == 1

    recovered.write_bytes(b"pdf")
    third = scanner.scan(name="Local test share", root_path=share)
    assert third.discovered == 0
    assert third.changed == 1
    assert third.unchanged == 2
    assert third.missing == 0
    with session_factory() as session:
        tracked = session.scalar(
            select(NetworkShareFile).where(NetworkShareFile.relative_path == "recovered.pdf")
        )
        assert tracked is not None
        assert tracked.status is NetworkShareFileStatus.DISCOVERED


def test_unavailable_share_does_not_mark_known_files_missing(
    tmp_path: Path,
    session_factory: sessionmaker[Session],
) -> None:
    share = tmp_path / "share"
    share.mkdir()
    (share / "notes.txt").write_text("hello", encoding="utf-8")
    scanner = NetworkShareScanner(session_factory=session_factory)
    scanner.scan(name="Local test share", root_path=share)
    share.rename(tmp_path / "offline")

    with pytest.raises(NetworkShareUnavailableError, match="does not exist"):
        scanner.scan(name="Local test share", root_path=share)

    with session_factory() as session:
        source = session.scalar(select(NetworkShareSource))
        tracked = session.scalar(select(NetworkShareFile))
        assert source is not None
        assert source.status is NetworkShareStatus.UNAVAILABLE
        assert source.last_error is not None
        assert tracked is not None
        assert tracked.status is NetworkShareFileStatus.DISCOVERED
