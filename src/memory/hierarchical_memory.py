from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal


MemoryLayer = Literal[
    "object",
    "spatial",
    "task",
    "failure",
    "skill",
    "episode",
]

MEMORY_LAYERS: tuple[MemoryLayer, ...] = (
    "object",
    "spatial",
    "task",
    "failure",
    "skill",
    "episode",
)

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


@dataclass(frozen=True)
class EvidenceReference:
    session_id: str
    step_id: int
    source: str
    reference: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LayeredMemoryRecord:
    id: int
    layer: MemoryLayer
    identity_key: str
    session_id: str
    instruction: str
    subject: str
    summary: str
    success: bool | None
    confidence: float
    metadata: dict[str, Any]
    evidence: list[dict[str, Any]]
    occurrence_count: int
    access_count: int
    created_at: str
    updated_at: str
    last_accessed_at: str
    similarity: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["lesson"] = self.summary
        payload["action"] = self.metadata.get("action")
        payload["action_success"] = self.success
        payload["region"] = self.metadata.get("region")
        return payload


class HierarchicalMemoryStore:
    """Persistent six-layer memory with deterministic bounded retention."""

    def __init__(
        self,
        db_path: Path | str,
        *,
        capacity: int,
        failure_capacity: int,
    ):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if failure_capacity <= 0:
            raise ValueError("failure_capacity must be positive")
        self.db_path = Path(db_path)
        self.capacity = int(capacity)
        self.failure_capacity = int(failure_capacity)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def upsert(
        self,
        *,
        layer: MemoryLayer,
        identity_key: str,
        session_id: str,
        instruction: str,
        subject: str,
        summary: str,
        evidence: EvidenceReference | dict[str, Any],
        success: bool | None = None,
        confidence: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        self._validate_layer(layer)
        if not identity_key.strip():
            raise ValueError("identity_key is required")
        if not summary.strip():
            raise ValueError("summary is required")
        normalized_evidence = self._normalize_evidence(evidence)
        identity_hash = hashlib.sha256(
            f"{layer}\0{identity_key}".encode("utf-8")
        ).hexdigest()
        now = datetime.now(timezone.utc).isoformat()

        with self._lock, self._connect() as connection:
            existing = connection.execute(
                """
                SELECT id, metadata_json, evidence_json
                FROM hierarchical_memories
                WHERE layer = ? AND identity_hash = ?
                """,
                (layer, identity_hash),
            ).fetchone()
            if existing is None:
                cursor = connection.execute(
                    """
                    INSERT INTO hierarchical_memories (
                        layer,
                        identity_hash,
                        identity_key,
                        session_id,
                        instruction,
                        subject,
                        summary,
                        success,
                        confidence,
                        metadata_json,
                        evidence_json,
                        occurrence_count,
                        access_count,
                        created_at,
                        updated_at,
                        last_accessed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 0, ?, ?, ?)
                    """,
                    (
                        layer,
                        identity_hash,
                        identity_key,
                        session_id,
                        instruction,
                        subject,
                        summary,
                        self._encode_success(success),
                        float(confidence),
                        self._dump_json(metadata or {}),
                        self._dump_json([normalized_evidence]),
                        now,
                        now,
                        now,
                    ),
                )
                memory_id = int(cursor.lastrowid)
            else:
                merged_metadata = json.loads(str(existing["metadata_json"]))
                merged_metadata.update(metadata or {})
                evidence_items = json.loads(str(existing["evidence_json"]))
                evidence_items = self._merge_evidence(
                    evidence_items,
                    normalized_evidence,
                )
                memory_id = int(existing["id"])
                connection.execute(
                    """
                    UPDATE hierarchical_memories
                    SET session_id = ?,
                        instruction = ?,
                        subject = ?,
                        summary = ?,
                        success = ?,
                        confidence = ?,
                        metadata_json = ?,
                        evidence_json = ?,
                        occurrence_count = occurrence_count + 1,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        session_id,
                        instruction,
                        subject,
                        summary,
                        self._encode_success(success),
                        float(confidence),
                        self._dump_json(merged_metadata),
                        self._dump_json(evidence_items),
                        now,
                        memory_id,
                    ),
                )
            self._prune(connection)
        return memory_id

    def search(
        self,
        query: str,
        *,
        top_k: int,
        layers: list[MemoryLayer] | tuple[MemoryLayer, ...] | None = None,
        exclude_session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []
        selected_layers = tuple(layers or MEMORY_LAYERS)
        for layer in selected_layers:
            self._validate_layer(layer)
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        placeholders = ", ".join("?" for _ in selected_layers)
        sql = (
            "SELECT * FROM hierarchical_memories "
            f"WHERE layer IN ({placeholders})"
        )
        parameters: list[Any] = list(selected_layers)
        if exclude_session_id is not None:
            sql += " AND session_id != ?"
            parameters.append(exclude_session_id)

        with self._lock, self._connect() as connection:
            rows = connection.execute(sql, parameters).fetchall()
            ranked: list[
                tuple[tuple[float, float, int, int], LayeredMemoryRecord]
            ] = []
            for row in rows:
                record = self._row_to_record(row)
                searchable = " ".join(
                    (
                        record.instruction,
                        record.subject,
                        record.summary,
                        self._dump_json(record.metadata),
                    )
                )
                memory_tokens = _tokenize(searchable)
                union = query_tokens | memory_tokens
                similarity = (
                    len(query_tokens & memory_tokens) / len(union)
                    if union
                    else 0.0
                )
                if similarity <= 0.0:
                    continue
                record = LayeredMemoryRecord(
                    **{
                        **asdict(record),
                        "similarity": similarity,
                    }
                )
                ranked.append(
                    (
                        (
                            similarity,
                            record.confidence,
                            record.occurrence_count,
                            record.id,
                        ),
                        record,
                    )
                )
            ranked.sort(key=lambda item: item[0], reverse=True)
            selected = [record for _, record in ranked[:top_k]]
            if selected:
                now = datetime.now(timezone.utc).isoformat()
                connection.executemany(
                    """
                    UPDATE hierarchical_memories
                    SET access_count = access_count + 1,
                        last_accessed_at = ?
                    WHERE id = ?
                    """,
                    [(now, record.id) for record in selected],
                )
        results: list[dict[str, Any]] = []
        for record in selected:
            payload = record.to_dict()
            payload["evidence"] = payload["evidence"][-top_k:]
            results.append(payload)
        return results

    def search_grouped(
        self,
        query: str,
        *,
        top_k: int,
        exclude_session_id: str | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {
            layer: [] for layer in MEMORY_LAYERS
        }
        for record in self.search(
            query,
            top_k=top_k,
            layers=list(MEMORY_LAYERS),
            exclude_session_id=exclude_session_id,
        ):
            grouped[str(record["layer"])].append(record)
        return grouped

    def get(self, memory_id: int) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM hierarchical_memories WHERE id = ?",
                (int(memory_id),),
            ).fetchone()
        return self._row_to_record(row).to_dict() if row is not None else None

    def count(self, *, layer: MemoryLayer | None = None) -> int:
        if layer is not None:
            self._validate_layer(layer)
        with self._lock, self._connect() as connection:
            if layer is None:
                row = connection.execute(
                    "SELECT COUNT(*) AS count FROM hierarchical_memories"
                ).fetchone()
            else:
                row = connection.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM hierarchical_memories
                    WHERE layer = ?
                    """,
                    (layer,),
                ).fetchone()
        return int(row["count"])

    def layer_counts(self) -> dict[str, int]:
        return {layer: self.count(layer=layer) for layer in MEMORY_LAYERS}

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS hierarchical_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    layer TEXT NOT NULL,
                    identity_hash TEXT NOT NULL,
                    identity_key TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    instruction TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    success INTEGER,
                    confidence REAL NOT NULL,
                    metadata_json TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    occurrence_count INTEGER NOT NULL,
                    access_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_accessed_at TEXT NOT NULL,
                    UNIQUE(layer, identity_hash)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_hierarchical_layer_updated
                ON hierarchical_memories(layer, updated_at, id)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_hierarchical_session
                ON hierarchical_memories(session_id)
                """
            )

    def _prune(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            DELETE FROM hierarchical_memories
            WHERE layer = 'failure'
              AND id NOT IN (
                  SELECT id
                  FROM hierarchical_memories
                  WHERE layer = 'failure'
                  ORDER BY updated_at DESC, id DESC
                  LIMIT ?
              )
            """,
            (self.failure_capacity,),
        )
        connection.execute(
            """
            DELETE FROM hierarchical_memories
            WHERE id NOT IN (
                SELECT id
                FROM hierarchical_memories
                ORDER BY updated_at DESC, id DESC
                LIMIT ?
            )
            """,
            (self.capacity,),
        )

    def _merge_evidence(
        self,
        existing: list[dict[str, Any]],
        incoming: dict[str, Any],
    ) -> list[dict[str, Any]]:
        incoming_key = self._dump_json(incoming)
        merged = [
            item
            for item in existing
            if self._dump_json(item) != incoming_key
        ]
        merged.append(incoming)
        return merged[-self.capacity :]

    @staticmethod
    def _normalize_evidence(
        evidence: EvidenceReference | dict[str, Any],
    ) -> dict[str, Any]:
        payload = (
            evidence.to_dict()
            if isinstance(evidence, EvidenceReference)
            else dict(evidence)
        )
        required = {"session_id", "step_id", "source", "reference"}
        missing = required.difference(payload)
        if missing:
            raise ValueError(
                f"evidence is missing required fields: {sorted(missing)}"
            )
        payload["session_id"] = str(payload["session_id"])
        payload["step_id"] = int(payload["step_id"])
        payload["source"] = str(payload["source"])
        payload["reference"] = str(payload["reference"])
        payload["details"] = dict(payload.get("details") or {})
        return payload

    @staticmethod
    def _encode_success(success: bool | None) -> int | None:
        return None if success is None else int(bool(success))

    @staticmethod
    def _decode_success(success: Any) -> bool | None:
        return None if success is None else bool(success)

    @staticmethod
    def _dump_json(payload: Any) -> str:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _validate_layer(layer: str) -> None:
        if layer not in MEMORY_LAYERS:
            raise ValueError(f"unsupported memory layer: {layer}")

    def _row_to_record(self, row: sqlite3.Row) -> LayeredMemoryRecord:
        return LayeredMemoryRecord(
            id=int(row["id"]),
            layer=str(row["layer"]),
            identity_key=str(row["identity_key"]),
            session_id=str(row["session_id"]),
            instruction=str(row["instruction"]),
            subject=str(row["subject"]),
            summary=str(row["summary"]),
            success=self._decode_success(row["success"]),
            confidence=float(row["confidence"]),
            metadata=json.loads(str(row["metadata_json"])),
            evidence=json.loads(str(row["evidence_json"])),
            occurrence_count=int(row["occurrence_count"]),
            access_count=int(row["access_count"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            last_accessed_at=str(row["last_accessed_at"]),
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


def normalize_identity(value: Any) -> str:
    if isinstance(value, str):
        return " ".join(value.lower().split())
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _tokenize(text: str) -> set[str]:
    normalized = text.lower()
    tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", normalized)
        if token not in _ENGLISH_STOPWORDS
    }
    for segment in re.findall(r"[\u4e00-\u9fff]+", normalized):
        if len(segment) == 1:
            tokens.add(segment)
        else:
            tokens.update(
                segment[index : index + 2]
                for index in range(len(segment) - 1)
            )
    return tokens
