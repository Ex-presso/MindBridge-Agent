"""Destructive memory migrations must preserve logical-deletion tombstones."""

import importlib.util
from pathlib import Path

import pytest

from app.core.agent.safety import CRISIS_DETECTOR_VERSION


def _load_migration(filename: str):
    path = Path(__file__).resolve().parents[1] / "alembic" / "versions" / filename
    spec = importlib.util.spec_from_file_location(f"test_{filename}", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Result:
    def __init__(self, value: bool) -> None:
        self.value = value

    def scalar_one(self) -> bool:
        return self.value


class _Bind:
    def __init__(self, value: bool) -> None:
        self.value = value
        self.sql: list[str] = []

    def execute(self, statement):
        self.sql.append(str(statement))
        return _Result(self.value)


@pytest.mark.parametrize("unsafe", [False, True])
def test_004_downgrade_rejects_non_default_memory_safety_state(unsafe: bool):
    migration = _load_migration("004_add_memory_write_safety.py")
    bind = _Bind(unsafe)

    if unsafe:
        with pytest.raises(RuntimeError, match="Cannot downgrade"):
            migration._assert_safe_downgrade(bind)
    else:
        migration._assert_safe_downgrade(bind)

    assert "LOCK TABLE users, conversations" in bind.sql[0]
    assert "IN EXCLUSIVE MODE" in bind.sql[0]
    assert "memory_data_epoch" in bind.sql[-1]
    assert "memory_crisis_seen" in bind.sql[-1]
    assert "memory_jobs" in bind.sql[-1]


@pytest.mark.parametrize("pending", [False, True])
def test_005_downgrade_preserves_account_and_crisis_review_gates(pending: bool):
    migration = _load_migration("005_harden_memory_ownership.py")
    bind = _Bind(pending)

    if pending:
        with pytest.raises(RuntimeError, match="historical crisis review"):
            migration._assert_safe_downgrade(bind)
    else:
        migration._assert_safe_downgrade(bind)

    assert "LOCK TABLE users, conversations" in bind.sql[0]
    assert "IN EXCLUSIVE MODE" in bind.sql[0]
    assert "account_deletion_pending" in bind.sql[-1]
    assert "memory_crisis_reviewed" in bind.sql[-1]
    assert "memory_crisis_review_version" in bind.sql[-1]
    # Migration 005 froze the detector version it shipped with. Every live
    # bump must be recorded here after re-checking the downgrade guard: rows
    # reviewed at a newer version make the preflight fail closed, which is
    # the intended behavior. Reviewed bumps: 3 -> 4 (English-only detection).
    assert migration.CRISIS_DETECTOR_VERSION == 3
    assert CRISIS_DETECTOR_VERSION == 5
