from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


_ENGLISH_STOPWORDS = {
    "a",
    "an",
    "find",
    "for",
    "locate",
    "object",
    "search",
    "target",
    "the",
}
_CHINESE_STOP_PHRASES = (
    "请",
    "一个",
    "这个",
    "寻找",
    "搜索",
    "查找",
    "找到",
    "目标",
    "物体",
)


@dataclass(frozen=True)
class EpisodicMemory:
    id: int
    namespace: str
    session_id: str
    instruction: str
    action: str
    action_success: bool
    confidence: float
    region: str | None
    lesson: str
    metadata: dict[str, Any]
    created_at: str
    similarity: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EpisodicMemoryStore:
    """Persistent execution memory with deterministic task-similarity retrieval."""

    def __init__(self, db_path: Path | str, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.db_path = Path(db_path)
        self.capacity = int(capacity)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def add(
        self,
        *,
        namespace: str,
        session_id: str,
        instruction: str,
        action: str,
        action_success: bool,
        confidence: float,
        region: str | None,
        lesson: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO episodic_memories (
                    namespace,
                    session_id,
                    instruction,
                    action,
                    action_success,
                    confidence,
                    region,
                    lesson,
                    metadata_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    namespace,
                    session_id,
                    instruction,
                    action,
                    int(bool(action_success)),
                    float(confidence),
                    region,
                    lesson,
                    json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
                    created_at,
                ),
            )
            memory_id = int(cursor.lastrowid)
            connection.execute(
                """
                DELETE FROM episodic_memories
                WHERE namespace = ?
                  AND id NOT IN (
                      SELECT id
                      FROM episodic_memories
                      WHERE namespace = ?
                      ORDER BY id DESC
                      LIMIT ?
                  )
                """,
                (namespace, namespace, self.capacity),
            )
        return memory_id

    def search(
        self,
        query: str,
        *,
        namespace: str,
        top_k: int,
        exclude_session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        sql = """
            SELECT
                id,
                namespace,
                session_id,
                instruction,
                action,
                action_success,
                confidence,
                region,
                lesson,
                metadata_json,
                created_at
            FROM episodic_memories
            WHERE namespace = ?
        """
        parameters: list[Any] = [namespace]
        if exclude_session_id is not None:
            sql += " AND session_id != ?"
            parameters.append(exclude_session_id)

        with self._connect() as connection:
            rows = connection.execute(sql, parameters).fetchall()

        ranked: list[tuple[tuple[float, int, float, int], EpisodicMemory]] = []
        for row in rows:
            memory_tokens = _tokenize(str(row["instruction"]))
            union = query_tokens | memory_tokens
            similarity = len(query_tokens & memory_tokens) / len(union) if union else 0.0
            if similarity <= 0.0:
                continue
            memory = EpisodicMemory(
                id=int(row["id"]),
                namespace=str(row["namespace"]),
                session_id=str(row["session_id"]),
                instruction=str(row["instruction"]),
                action=str(row["action"]),
                action_success=bool(row["action_success"]),
                confidence=float(row["confidence"]),
                region=str(row["region"]) if row["region"] is not None else None,
                lesson=str(row["lesson"]),
                metadata=json.loads(str(row["metadata_json"])),
                created_at=str(row["created_at"]),
                similarity=similarity,
            )
            rank = (
                similarity,
                int(memory.action_success),
                memory.confidence,
                memory.id,
            )
            ranked.append((rank, memory))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [memory.to_dict() for _, memory in ranked[:top_k]]

    def count(self, *, namespace: str | None = None) -> int:
        with self._connect() as connection:
            if namespace is None:
                row = connection.execute(
                    "SELECT COUNT(*) AS count FROM episodic_memories"
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT COUNT(*) AS count FROM episodic_memories WHERE namespace = ?",
                    (namespace,),
                ).fetchone()
        return int(row["count"])

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS episodic_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    instruction TEXT NOT NULL,
                    action TEXT NOT NULL,
                    action_success INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    region TEXT,
                    lesson TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_episodic_namespace_session
                ON episodic_memories(namespace, session_id)
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


def _tokenize(text: str) -> set[str]:
    normalized = text.lower()
    for phrase in _CHINESE_STOP_PHRASES:
        normalized = normalized.replace(phrase, " ")

    tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", normalized)
        if token not in _ENGLISH_STOPWORDS
    }
    for segment in re.findall(r"[\u4e00-\u9fff]+", normalized):
        if len(segment) == 1:
            tokens.add(segment)
        else:
            tokens.update(segment[index : index + 2] for index in range(len(segment) - 1))
    return tokens
