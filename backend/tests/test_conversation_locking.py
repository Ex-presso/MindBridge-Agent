"""Conversation generation/deletion use a PostgreSQL row lock."""

import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

from sqlalchemy.dialects import postgresql

from app.db.repositories import conversation_repo


def test_owned_conversation_query_uses_for_update():
    conversation_id = uuid.uuid4()
    user_id = uuid.uuid4()
    expected = SimpleNamespace(id=conversation_id, user_id=user_id)
    result = SimpleNamespace(scalar_one_or_none=lambda: expected)
    db = SimpleNamespace(execute=AsyncMock(return_value=result))

    found = asyncio.run(
        conversation_repo.get_owned_for_update(db, conversation_id, user_id)
    )

    assert found is expected
    statement = db.execute.await_args.args[0]
    sql = str(statement.compile(dialect=postgresql.dialect()))
    assert "FOR UPDATE" in sql
    assert statement.compile().params == {
        "id_1": conversation_id,
        "user_id_1": user_id,
    }
