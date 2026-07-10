"""Safe Alembic adoption and upgrade for application-owned tables.

Historically, local and Docker startup used ``Base.metadata.create_all``. Those
databases contain the application tables but no ``alembic_version`` row. This
module recognizes only the known legacy layouts, adopts the matching Alembic
revision, and then upgrades normally. Unknown partial layouts fail closed.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path

import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncEngine

from config.settings import settings

logger = logging.getLogger(__name__)

CORE_TABLE_COLUMNS: dict[str, frozenset[str]] = {
    "users": frozenset(
        {"id", "email", "hashed_password", "display_name", "created_at"}
    ),
    "conversations": frozenset(
        {
            "id",
            "user_id",
            "title",
            "model",
            "provider",
            "created_at",
            "updated_at",
        }
    ),
    "messages": frozenset(
        {
            "id",
            "conversation_id",
            "role",
            "content",
            "model_used",
            "provider",
            "tokens_used",
            "created_at",
        }
    ),
}

API_KEY_COLUMNS = frozenset(
    {
        "id",
        "user_id",
        "provider",
        "api_key_encrypted",
        "base_url",
        "model_id",
        "display_name",
        "created_at",
        "updated_at",
    }
)

APPLICATION_TABLES = frozenset({*CORE_TABLE_COLUMNS, "user_api_keys"})

# Transaction-scoped PostgreSQL advisory lock. It serializes startup migration
# attempts across multiple app processes without requiring a persistent lock.
SCHEMA_MIGRATION_LOCK_ID = 5577986816715778404


class LegacySchemaError(RuntimeError):
    """Raised when an unversioned schema cannot be adopted safely."""


@dataclass(frozen=True)
class SchemaUpgradePlan:
    """Decision made before invoking Alembic."""

    adopt_revision: str | None
    reason: str


def decide_schema_upgrade(
    *,
    current_revisions: Collection[str],
    existing_tables: Collection[str],
    columns_by_table: Mapping[str, Collection[str]],
) -> SchemaUpgradePlan:
    """Choose a safe adoption point for an existing database.

    Versioned databases are always left to Alembic. An unversioned database is
    either fresh, a complete known ``create_all`` layout, or an error. The
    memory column distinguishes the two legacy baselines currently supported.
    """
    if current_revisions:
        return SchemaUpgradePlan(
            adopt_revision=None,
            reason="database already has an Alembic revision",
        )

    tables = set(existing_tables)
    present_application_tables = tables & APPLICATION_TABLES
    if not present_application_tables:
        return SchemaUpgradePlan(
            adopt_revision=None,
            reason="fresh database",
        )

    core_tables = set(CORE_TABLE_COLUMNS)
    if not core_tables.issubset(present_application_tables):
        missing = sorted(core_tables - present_application_tables)
        present = sorted(present_application_tables)
        raise LegacySchemaError(
            "Cannot adopt partial unversioned schema: "
            f"found application tables {present}; missing core tables {missing}. "
            "Restore the missing tables or migrate the database manually."
        )

    expected_columns = dict(CORE_TABLE_COLUMNS)
    if "user_api_keys" in present_application_tables:
        expected_columns["user_api_keys"] = API_KEY_COLUMNS

    for table_name, required_columns in expected_columns.items():
        actual_columns = set(columns_by_table.get(table_name, ()))
        missing_columns = sorted(required_columns - actual_columns)
        if missing_columns:
            raise LegacySchemaError(
                "Cannot adopt incompatible unversioned schema: "
                f"table {table_name!r} is missing columns {missing_columns}. "
                "Migrate the database manually."
            )

    users_columns = set(columns_by_table.get("users", ()))
    revision = "002" if "memory_enabled" in users_columns else "001"
    return SchemaUpgradePlan(
        adopt_revision=revision,
        reason=f"recognized legacy create_all schema at revision {revision}",
    )


def make_alembic_config(*, connection: Connection | None = None) -> Config:
    """Build an Alembic config that works independently of the current cwd."""
    project_root = Path(settings.PROJECT_ROOT)
    config = Config(str(project_root / "alembic.ini"))
    config.set_main_option("script_location", str(project_root / "alembic"))
    config.attributes["configure_logger"] = False
    if connection is not None:
        config.attributes["connection"] = connection
    return config


def upgrade_schema(connection: Connection) -> SchemaUpgradePlan:
    """Adopt a recognized legacy schema if needed, then upgrade to head."""
    connection.execute(
        sa.text("SELECT pg_advisory_xact_lock(:lock_id)"),
        {"lock_id": SCHEMA_MIGRATION_LOCK_ID},
    )

    inspector = sa.inspect(connection)
    existing_tables = set(inspector.get_table_names())
    application_columns = {
        table_name: {
            column["name"] for column in inspector.get_columns(table_name)
        }
        for table_name in existing_tables & APPLICATION_TABLES
    }
    current_revisions = MigrationContext.configure(connection).get_current_heads()
    plan = decide_schema_upgrade(
        current_revisions=current_revisions,
        existing_tables=existing_tables,
        columns_by_table=application_columns,
    )

    config = make_alembic_config(connection=connection)
    if plan.adopt_revision is not None:
        logger.warning("Adopting %s", plan.reason)
        command.stamp(config, plan.adopt_revision)

    command.upgrade(config, "head")
    return plan


async def ensure_schema_at_head(engine: AsyncEngine) -> SchemaUpgradePlan:
    """Run the safe adoption-and-upgrade flow on an async SQLAlchemy engine."""
    async with engine.begin() as connection:
        return await connection.run_sync(upgrade_schema)
