"""Explicit, configuration-driven database connections and session lifecycle."""

import os
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import ArgumentError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from retention.utils.logger import get_logger

logger = get_logger(__name__)


def _same_database(left, right) -> bool:
    """Conservatively reject reuse of an application database name."""
    return left.database == right.database


def database_url(*, test: bool = False) -> str:
    key = "TEST_DATABASE_URL" if test else "DATABASE_URL"
    value = os.getenv(key, "")
    if not value:
        raise ValueError(f"{key} is required")
    try:
        url = make_url(value)
    except ArgumentError as exc:
        raise ValueError(f"{key} is not a valid database URL") from exc
    if url.drivername != "postgresql+psycopg" or not url.database:
        raise ValueError(f"{key} must use postgresql+psycopg and name a database")
    if test:
        app_url = os.getenv("DATABASE_URL")
        if app_url and _same_database(make_url(app_url), url):
            raise ValueError("Test database must differ from application database")
        if not url.database.endswith("_test"):
            raise ValueError("Test database name must end with _test")
    return value


def create_database_engine(url: str | None = None, *, test: bool = False) -> Engine:
    value = url if url is not None else database_url(test=test)
    try:
        parsed = make_url(value)
    except ArgumentError as exc:
        raise ValueError("Invalid database URL") from exc
    if parsed.drivername != "postgresql+psycopg" or not parsed.database:
        raise ValueError("A postgresql+psycopg database URL is required")
    if test and (
        not parsed.database.endswith("_test")
        or (
            os.getenv("DATABASE_URL")
            and _same_database(parsed, make_url(os.environ["DATABASE_URL"]))
        )
    ):
        raise ValueError("Test engine must target a separate _test database")
    engine = create_engine(
        value, pool_pre_ping=True, pool_size=5, max_overflow=5, pool_recycle=1800
    )
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        engine.dispose()
        logger.error("Database connection failed")
        raise ConnectionError("Database connection failed") from exc
    return engine


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """Commit on success, rollback on error, and always close the session."""
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
