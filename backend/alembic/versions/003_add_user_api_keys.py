"""add user API keys

Revision ID: 003
Revises: 002
Create Date: 2026-07-10 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

REQUIRED_COLUMNS = {
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


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if inspector.has_table("user_api_keys"):
        existing_columns = {
            column["name"] for column in inspector.get_columns("user_api_keys")
        }
        missing_columns = sorted(REQUIRED_COLUMNS - existing_columns)
        if missing_columns:
            raise RuntimeError(
                "Existing user_api_keys table is incompatible; "
                f"missing columns: {missing_columns}"
            )
    else:
        op.create_table(
            "user_api_keys",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("user_id", sa.Uuid(), nullable=False),
            sa.Column("provider", sa.String(50), nullable=False),
            sa.Column("api_key_encrypted", sa.Text(), nullable=False),
            sa.Column("base_url", sa.String(500), nullable=True),
            sa.Column("model_id", sa.String(100), nullable=True),
            sa.Column("display_name", sa.String(100), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )

    inspector = sa.inspect(bind)
    user_id_is_indexed = any(
        index.get("column_names") == ["user_id"]
        for index in inspector.get_indexes("user_api_keys")
    )
    if not user_id_is_indexed:
        op.create_index(
            op.f("ix_user_api_keys_user_id"),
            "user_api_keys",
            ["user_id"],
            unique=False,
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("user_api_keys"):
        return

    index_names = {
        index["name"] for index in inspector.get_indexes("user_api_keys")
    }
    index_name = op.f("ix_user_api_keys_user_id")
    if index_name in index_names:
        op.drop_index(index_name, table_name="user_api_keys")
    op.drop_table("user_api_keys")
