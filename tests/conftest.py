"""Shared test fixtures.

DB-dependent tests are skipped automatically if no Postgres is reachable at
config.Settings.database_url — this keeps `pytest` runnable with zero
infrastructure for pure-logic tests (splitter, reducer, ranker MMR), while
still exercising the real schema against a real Postgres when docker-compose
is up (`docker-compose up -d postgres redis` before running the suite).
"""

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from models.db import Base, get_engine, get_session_factory
from orchestrator.celery_app import celery_app


def _postgres_reachable() -> bool:
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except OperationalError:
        return False


@pytest.fixture(scope="session")
def postgres_available() -> bool:
    return _postgres_reachable()


@pytest.fixture
def db_session(postgres_available):
    if not postgres_available:
        pytest.skip("Postgres not reachable at config.Settings.database_url — start docker-compose to run this test")

    Base.metadata.create_all(get_engine())
    session = get_session_factory()()
    try:
        yield session
        session.rollback()
    finally:
        for table in reversed(Base.metadata.sorted_tables):
            session.execute(table.delete())
        session.commit()
        session.close()


@pytest.fixture(autouse=True, scope="session")
def _celery_eager_mode():
    """All Celery tasks run synchronously in-process during tests — no
    broker/worker required to test task logic or chain/chord wiring."""
    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = True
    yield
