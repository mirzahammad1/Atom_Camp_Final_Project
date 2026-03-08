from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Dict, Optional

from pymongo import MongoClient

logger = logging.getLogger("eduassist.response_log")


class ResponseLogAgent:
    """
    Logs every user query and system response to MongoDB.

    Stores:
    - user query (original)
    - final answer
    - source (FAQ / VECTOR / WEB / Tool)
    - intent type (RAG / ACTION / HYBRID)
    - consistency passed flag
    - timestamp

    ✅ Non-blocking — uses background thread to avoid slowing responses
    ✅ Thread-safe
    ✅ Fails silently — never crashes the main pipeline
    """

    COLLECTION_NAME = "response_logs"

    def __init__(self, mongo_uri: str, db_name: str):
        self._lock = threading.Lock()
        try:
            self._client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
            self._collection = self._client[db_name][self.COLLECTION_NAME]
            # Create index on timestamp for efficient querying
            self._collection.create_index("timestamp")
            self._collection.create_index("intent_type")
            logger.info("✅ ResponseLogAgent initialized — logging to '%s.%s'", db_name, self.COLLECTION_NAME)
            self._enabled = True
        except Exception as e:
            logger.warning("⚠️ ResponseLogAgent disabled — MongoDB connection failed: %s", e)
            self._enabled = False

    def log(
        self,
        query: str,
        answer: str,
        source: str,
        intent_type: str = "RAG",
        consistency_passed: bool = True,
        consistency_attempts: int = 0,
        extra: Optional[Dict] = None,
    ) -> None:
        """
        Log a query-response pair asynchronously.

        Args:
            query:                 Original user question
            answer:                Final answer returned to user
            source:                Which agent answered (FAQ, VECTOR, WEB, Tool)
            intent_type:           RAG | ACTION | HYBRID | SMALLTALK | FALLBACK
            consistency_passed:    Whether consistency check passed
            consistency_attempts:  Number of consistency retry attempts
            extra:                 Any extra metadata dict
        """
        if not self._enabled:
            return

        # Log in background thread — non-blocking
        threading.Thread(
            target=self._write,
            args=(query, answer, source, intent_type, consistency_passed, consistency_attempts, extra),
            daemon=True,
        ).start()

    def _write(
        self,
        query: str,
        answer: str,
        source: str,
        intent_type: str,
        consistency_passed: bool,
        consistency_attempts: int,
        extra: Optional[Dict],
    ) -> None:
        try:
            doc = {
                "timestamp":             datetime.now(timezone.utc),
                "query":                 (query or "").strip(),
                "answer":                (answer or "").strip(),
                "source":                source,
                "intent_type":           intent_type,
                "consistency_passed":    consistency_passed,
                "consistency_attempts":  consistency_attempts,
                "answer_length":         len((answer or "").strip()),
                "query_length":          len((query or "").strip()),
            }
            if extra and isinstance(extra, dict):
                doc.update(extra)

            with self._lock:
                self._collection.insert_one(doc)

            logger.debug("📝 Logged response — intent=%s source=%s", intent_type, source)

        except Exception as e:
            logger.warning("ResponseLog write error: %s", e)

    def get_recent(self, limit: int = 20) -> list:
        """Retrieve recent logs — useful for admin dashboard."""
        if not self._enabled:
            return []
        try:
            return list(
                self._collection
                .find({}, {"_id": 0})
                .sort("timestamp", -1)
                .limit(limit)
            )
        except Exception as e:
            logger.warning("ResponseLog read error: %s", e)
            return []

    def get_stats(self) -> Dict:
        """Return basic usage statistics."""
        if not self._enabled:
            return {}
        try:
            total = self._collection.count_documents({})
            by_intent = {}
            for intent in ["RAG", "ACTION", "HYBRID", "SMALLTALK", "FALLBACK"]:
                by_intent[intent] = self._collection.count_documents({"intent_type": intent})
            by_source = {}
            for source in ["FAQ Agent", "VECTOR Agent -> LLM", "WEB Agent -> LLM",
                           "Email Draft Tool", "Document Generator Tool", "Academic Info Tool"]:
                count = self._collection.count_documents({"source": {"$regex": source}})
                if count:
                    by_source[source] = count
            return {
                "total_queries": total,
                "by_intent":     by_intent,
                "by_source":     by_source,
            }
        except Exception as e:
            logger.warning("ResponseLog stats error: %s", e)
            return {}

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass