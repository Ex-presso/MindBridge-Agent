"""Pure tests for safe schema adoption and the Alembic revision graph."""

import pytest
from alembic.script import ScriptDirectory

from app.db.schema_migrations import (
    API_KEY_COLUMNS,
    CORE_TABLE_COLUMNS,
    LegacySchemaError,
    MEMORY_JOB_COLUMNS,
    WRITE_SAFETY_COLUMNS,
    decide_schema_upgrade,
    make_alembic_config,
)


def _legacy_columns(*, memory_enabled: bool, api_keys: bool = True):
    columns = {name: set(values) for name, values in CORE_TABLE_COLUMNS.items()}
    if memory_enabled:
        columns["users"].add("memory_enabled")
    if api_keys:
        columns["user_api_keys"] = set(API_KEY_COLUMNS)
    return columns


def test_fresh_database_runs_full_migration_chain_without_adoption():
    plan = decide_schema_upgrade(
        current_revisions=(),
        existing_tables={"langgraph_checkpoints"},
        columns_by_table={},
    )

    assert plan.adopt_revision is None
    assert plan.reason == "fresh database"


def test_legacy_create_all_database_without_memory_adopts_001():
    columns = _legacy_columns(memory_enabled=False)

    plan = decide_schema_upgrade(
        current_revisions=(),
        existing_tables=columns,
        columns_by_table=columns,
    )

    assert plan.adopt_revision == "001"


def test_current_create_all_database_with_memory_adopts_002():
    columns = _legacy_columns(memory_enabled=True)

    plan = decide_schema_upgrade(
        current_revisions=(),
        existing_tables=columns,
        columns_by_table=columns,
    )

    assert plan.adopt_revision == "002"


def test_current_write_safety_create_all_database_adopts_004():
    columns = _legacy_columns(memory_enabled=True)
    columns["users"].update(WRITE_SAFETY_COLUMNS["users"])
    columns["conversations"].update(WRITE_SAFETY_COLUMNS["conversations"])
    columns["memory_jobs"] = set(MEMORY_JOB_COLUMNS)

    plan = decide_schema_upgrade(
        current_revisions=(),
        existing_tables=columns,
        columns_by_table=columns,
    )

    assert plan.adopt_revision == "004"


def test_partial_write_safety_create_all_database_fails_closed():
    columns = _legacy_columns(memory_enabled=True)
    columns["users"].add("memory_data_epoch")

    with pytest.raises(LegacySchemaError, match="write-safety schema"):
        decide_schema_upgrade(
            current_revisions=(),
            existing_tables=columns,
            columns_by_table=columns,
        )


def test_write_safety_schema_without_consent_column_fails_closed():
    columns = _legacy_columns(memory_enabled=False)
    columns["users"].update(WRITE_SAFETY_COLUMNS["users"])
    columns["conversations"].update(WRITE_SAFETY_COLUMNS["conversations"])
    columns["memory_jobs"] = set(MEMORY_JOB_COLUMNS)

    with pytest.raises(LegacySchemaError, match="memory_enabled"):
        decide_schema_upgrade(
            current_revisions=(),
            existing_tables=columns,
            columns_by_table=columns,
        )


@pytest.mark.parametrize("revision", ["001", "002", "003", "004"])
def test_versioned_database_never_gets_restamped(revision: str):
    plan = decide_schema_upgrade(
        current_revisions=(revision,),
        existing_tables={"users"},
        columns_by_table={"users": {"id"}},
    )

    assert plan.adopt_revision is None


def test_legacy_database_without_api_key_table_is_supported():
    columns = _legacy_columns(memory_enabled=False, api_keys=False)

    plan = decide_schema_upgrade(
        current_revisions=(),
        existing_tables=columns,
        columns_by_table=columns,
    )

    assert plan.adopt_revision == "001"


def test_partial_unversioned_schema_fails_closed():
    with pytest.raises(LegacySchemaError, match="partial unversioned schema"):
        decide_schema_upgrade(
            current_revisions=(),
            existing_tables={"users", "conversations"},
            columns_by_table={
                "users": CORE_TABLE_COLUMNS["users"],
                "conversations": CORE_TABLE_COLUMNS["conversations"],
            },
        )


def test_incompatible_unversioned_schema_fails_closed():
    columns = _legacy_columns(memory_enabled=False)
    columns["messages"].remove("content")

    with pytest.raises(LegacySchemaError, match="missing columns.*content"):
        decide_schema_upgrade(
            current_revisions=(),
            existing_tables=columns,
            columns_by_table=columns,
        )


def test_alembic_revision_chain_has_single_004_head():
    scripts = ScriptDirectory.from_config(make_alembic_config())

    assert scripts.get_current_head() == "004"
