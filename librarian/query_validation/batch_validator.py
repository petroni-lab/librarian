"""Batch interface for validating multiple Europe PMC queries."""

from datetime import datetime, timezone
from typing import Any, Dict, List

from .validation import ValidationService


class BatchQueryValidator:
    def __init__(self, validation_service: ValidationService | None = None):
        self.validation_service = validation_service or ValidationService()

    def validate_queries(self, queries: List[str]) -> Dict[str, Any]:
        if queries is None:
            raise TypeError("queries must be a list of strings")
        if not isinstance(queries, list):
            raise TypeError("queries must be a list of strings")

        if not queries:
            return {
                "overall_status": "fail",
                "summary": {
                    "total_queries": 0,
                    "passed_queries": 0,
                    "failed_queries": 0,
                },
                "generated_at_utc": self._timestamp(),
                "results": [],
                "errors": ["No queries provided"],
            }

        results: List[Dict[str, Any]] = []
        passed_queries = 0

        for index, query in enumerate(queries):
            if not isinstance(query, str):
                raise TypeError(f"queries[{index}] must be a string")

            result = self.validation_service.validate_query(query)
            results.append({"index": index, **result})
            if result["valid"]:
                passed_queries += 1

        failed_queries = len(results) - passed_queries

        return {
            "overall_status": "pass" if failed_queries == 0 else "fail",
            "summary": {
                "total_queries": len(results),
                "passed_queries": passed_queries,
                "failed_queries": failed_queries,
            },
            "generated_at_utc": self._timestamp(),
            "results": results,
        }

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
