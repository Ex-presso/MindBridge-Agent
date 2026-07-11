"""Migration 004 must distinguish full and partial API-key uniqueness."""

import importlib.util
from pathlib import Path


def _load_migration_004():
    path = (
        Path(__file__).resolve().parents[1]
        / "alembic"
        / "versions"
        / "004_add_memory_write_safety.py"
    )
    spec = importlib.util.spec_from_file_location(
        "test_migration_004_add_memory_write_safety",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Inspector:
    def __init__(self, *, constraints=(), indexes=()):
        self.constraints = list(constraints)
        self.indexes = list(indexes)

    def get_unique_constraints(self, table_name):
        assert table_name == "user_api_keys"
        return self.constraints

    def get_indexes(self, table_name):
        assert table_name == "user_api_keys"
        return self.indexes


def test_partial_unique_index_does_not_satisfy_upsert_invariant():
    migration = _load_migration_004()
    inspector = _Inspector(
        indexes=[
            {
                "name": "uq_user_api_keys_active_provider",
                "unique": True,
                "column_names": ["user_id", "provider"],
                "dialect_options": {
                    "postgresql_where": "revoked_at IS NULL",
                },
            }
        ]
    )

    assert migration._has_user_provider_uniqueness(inspector) is False


def test_full_unique_index_satisfies_upsert_invariant():
    migration = _load_migration_004()
    inspector = _Inspector(
        indexes=[
            {
                "name": "uq_user_api_keys_provider",
                "unique": True,
                "column_names": ["provider", "user_id"],
                "dialect_options": {},
            }
        ]
    )

    assert migration._has_user_provider_uniqueness(inspector) is True


def test_differently_named_unique_constraint_satisfies_upsert_invariant():
    migration = _load_migration_004()
    inspector = _Inspector(
        constraints=[
            {
                "name": "legacy_unique_provider_per_user",
                "column_names": ["user_id", "provider"],
            }
        ]
    )

    assert migration._has_user_provider_uniqueness(inspector) is True
