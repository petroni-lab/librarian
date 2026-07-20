"""Deterministic validation logic, adapted to run independently."""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .epmc_rules import (
    BOOLEAN_FIELDS,
    DATE_FIELDS,
    FIXED_VALUE_FIELDS,
    LANGUAGE_CODE_ALIASES,
    LICENSE_CANONICAL_VALUES,
    VALID_EPMC_FIELDS,
    YEAR_ONLY_DATE_FIELDS,
)
from .query_parser import FieldClauseNode, ParseResult, TokenType, parse_query


ORCID_PATTERN = re.compile(r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$")
ISSN_PATTERN = re.compile(r"^\d{4}-\d{3}[\dX]$", re.IGNORECASE)
DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
BOOK_ID_PATTERN = re.compile(r"^NBK\d+$", re.IGNORECASE)
AUTH_MAN_ID_PATTERN = re.compile(r"^[A-Za-z]{2,}[A-Za-z0-9._-]*\d+[A-Za-z0-9._-]*$")
CITATION_LINK_PATTERN = re.compile(r"^[A-Za-z0-9.-]+_([A-Za-z]+)$")
PMCID_PATTERN = re.compile(r"^PMC\d+$", re.IGNORECASE)
YEAR_PATTERN = re.compile(r"^\d{4}$")
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
PURE_WILDCARD_RESTRICTED_FIELDS = {
    "GENE_PROTEIN",
    "DISEASE",
    "ORGANISM",
    "CHEBITERM",
    "GOTERM",
    "CHEM",
    "MESH",
}


class ValidationService:
    def validate_query(self, query: str) -> Dict[str, Any]:
        if not query or not query.strip():
            return self._failure_result(query, ["Query is empty or None"], [])

        parse_result = parse_query(query)
        errors: List[str] = list(parse_result.errors)
        warnings: List[str] = []

        self._validate_quote_tokens(parse_result, errors)
        self._validate_field_clauses(query, parse_result, errors, warnings)

        if not parse_result.root:
            errors.append("Query does not contain any searchable terms")

        errors = self._dedupe(errors)
        warnings = self._dedupe(warnings)

        if errors:
            return self._failure_result(query, errors, warnings)

        reason = "Deterministic syntax, field, date, and identifier validations passed"
        return {
            "query": query,
            "status": "pass",
            "valid": True,
            "reasons": [reason],
            "warnings": warnings,
        }

    def _failure_result(
        self, query: str, errors: List[str], warnings: List[str]
    ) -> Dict[str, Any]:
        unique_errors = self._dedupe(errors)
        unique_warnings = self._dedupe(warnings)
        return {
            "query": query,
            "status": "fail",
            "valid": False,
            "reasons": unique_errors,
            "warnings": unique_warnings,
        }

    def _validate_quote_tokens(
        self, parse_result: ParseResult, errors: List[str]
    ) -> None:
        for token in parse_result.tokens:
            if token.type == TokenType.QUOTED and (
                len(token.text) < 2 or not token.text.endswith('"')
            ):
                errors.append("Unmatched double quotes in query")
            if token.type == TokenType.SINGLE_QUOTED and (
                len(token.text) < 2 or not token.text.endswith("'")
            ):
                errors.append("Unmatched single quotes in query")

            if token.type in {TokenType.QUOTED, TokenType.SINGLE_QUOTED}:
                inner = token.text[1:-1] if len(token.text) >= 2 else token.text
                if "*" in inner or "?" in inner:
                    errors.append(f"Wildcard not allowed in quotes: {token.text}")

    def _validate_field_clauses(
        self,
        query: str,
        parse_result: ParseResult,
        errors: List[str],
        warnings: List[str],
    ) -> None:
        present_fields = {clause.field for clause in parse_result.field_clauses}

        for clause in parse_result.field_clauses:
            field = clause.field
            raw_field = clause.raw_field

            self._validate_field_tag(field, raw_field, errors)
            self._validate_colon_spacing(query, clause, errors)

            if field not in VALID_EPMC_FIELDS:
                continue

            if clause.value.kind == "missing" or not clause.value.raw.strip():
                errors.append(f"Field tag '{raw_field}' is missing a value")
                continue
            if (
                clause.value.kind in {"parenthesized", "range"}
                and not clause.value.raw[1:-1].strip()
            ):
                errors.append(f"Field tag '{raw_field}' is missing a value")
                continue

            normalized = self._normalize_value(clause.value.raw)

            if clause.value.kind == "bare" and self._contains_unquoted_whitespace(
                normalized
            ):
                errors.append(
                    f"Invalid syntax: {raw_field}:{clause.value.raw} - multi-word field values must use quotes or parentheses"
                )

            if field in BOOLEAN_FIELDS:
                self._validate_boolean_field(field, normalized, errors)

            if field == "SRC":
                self._validate_fixed_choice(
                    field, normalized.upper(), FIXED_VALUE_FIELDS["SRC"], errors
                )
            elif field == "ACCESSION_TYPE":
                self._validate_fixed_choice(
                    field,
                    normalized.lower(),
                    {value.lower() for value in FIXED_VALUE_FIELDS["ACCESSION_TYPE"]},
                    errors,
                )
            elif field == "ANNOTATION_TYPE":
                self._validate_annotation_type(normalized, errors)
            elif field == "AUTHORID_TYPE":
                self._validate_fixed_choice(
                    field, normalized.upper(), {"ORCID"}, errors
                )
            elif field == "AUTHOR_ROLES":
                self._validate_author_role(normalized, errors)
            elif field == "LANG":
                self._validate_language(normalized, errors)
            elif field == "LICENSE":
                self._validate_license(normalized, errors)

            if field in DATE_FIELDS:
                self._validate_date_field(field, normalized, clause.value.kind, errors)

            self._validate_field_specific_format(field, normalized, errors)
            self._validate_field_specific_wildcards(
                field, normalized, clause.value.kind, errors
            )

            if field == "GRANT_ID" and "GRANT_AGENCY" not in present_fields:
                warnings.append("GRANT_ID is usually used together with GRANT_AGENCY")
            if field == "EXT_ID" and "SRC" not in present_fields:
                warnings.append("EXT_ID is usually used together with SRC")
            if field == "INVESTIGATOR" and "AUTH_COLLECTIVE_LIST" not in present_fields:
                warnings.append(
                    "INVESTIGATOR is usually used together with AUTH_COLLECTIVE_LIST"
                )
            if field == "ISSUE" and (
                "JOURNAL" not in present_fields or "VOLUME" not in present_fields
            ):
                warnings.append(
                    "ISSUE is usually used together with JOURNAL and VOLUME"
                )

    def _validate_field_tag(
        self, field: str, raw_field: str, errors: List[str]
    ) -> None:
        if field in VALID_EPMC_FIELDS:
            return

        errors.append(f"Invalid field tag '{raw_field}'")

    def _validate_colon_spacing(
        self, query: str, clause: FieldClauseNode, errors: List[str]
    ) -> None:
        if clause.value.tokens:
            value_start = clause.value.tokens[0].start
        else:
            value_start = clause.start + len(clause.raw_field) + 1

        separator = query[clause.start : value_start]
        colon_index = separator.find(":")
        if colon_index == -1:
            return

        before_colon = separator[len(clause.raw_field) : colon_index]
        after_colon = separator[colon_index + 1 :]

        if before_colon != "":
            errors.append(
                f"Field tag '{clause.raw_field}' has invalid spacing before colon"
            )
        if after_colon and after_colon.strip() != after_colon:
            errors.append(
                f"Field tag '{clause.raw_field}' has invalid spacing after colon"
            )

    def _validate_boolean_field(
        self, field: str, normalized: str, errors: List[str]
    ) -> None:
        if normalized.upper() not in {"Y", "N"}:
            errors.append(
                f"Invalid value for {field}: '{normalized}' - must use Y or N"
            )

    def _validate_fixed_choice(
        self, field: str, normalized: str, allowed: set[str], errors: List[str]
    ) -> None:
        if normalized not in allowed:
            errors.append(f"Invalid value for {field}: '{normalized}'")

    def _validate_annotation_type(self, normalized: str, errors: List[str]) -> None:
        allowed = FIXED_VALUE_FIELDS["ANNOTATION_TYPE"]
        if normalized not in allowed:
            errors.append(f"Invalid value for ANNOTATION_TYPE: '{normalized}'")

    def _validate_author_role(self, normalized: str, errors: List[str]) -> None:
        allowed = {
            self._normalize_author_role(value)
            for value in FIXED_VALUE_FIELDS["AUTHOR_ROLES"]
        }
        if self._normalize_author_role(normalized) not in allowed:
            errors.append(f"Invalid value for AUTHOR_ROLES: '{normalized}'")

    def _validate_language(self, normalized: str, errors: List[str]) -> None:
        allowed_lower = {value.lower() for value in FIXED_VALUE_FIELDS["LANG"]}
        if normalized.lower() in allowed_lower:
            return

        alias = LANGUAGE_CODE_ALIASES.get(normalized.lower())
        if alias and alias in FIXED_VALUE_FIELDS["LANG"]:
            return

        errors.append(f"Invalid value for LANG: '{normalized}'")

    def _validate_license(self, normalized: str, errors: List[str]) -> None:
        canonical = self._canonicalize_license(normalized)
        if canonical not in LICENSE_CANONICAL_VALUES:
            errors.append(f"Invalid value for LICENSE: '{normalized}'")

    def _validate_date_field(
        self, field: str, normalized: str, value_kind: str, errors: List[str]
    ) -> None:
        if value_kind == "range":
            self._validate_date_range(field, normalized, errors)
            return

        if field in YEAR_ONLY_DATE_FIELDS:
            if not YEAR_PATTERN.fullmatch(normalized):
                errors.append(
                    f"Invalid value for {field}: '{normalized}' - expected YYYY"
                )
            return

        if not (YEAR_PATTERN.fullmatch(normalized) or self._is_valid_date(normalized)):
            errors.append(
                f"Invalid value for {field}: '{normalized}' - expected YYYY or YYYY-MM-DD"
            )

    def _validate_date_range(
        self, field: str, normalized: str, errors: List[str]
    ) -> None:
        if not normalized.startswith("[") or not normalized.endswith("]"):
            errors.append(f"Invalid range for {field}: '{normalized}'")
            return

        content = normalized[1:-1].strip()
        parts = re.split(r"\s+TO\s+", content, flags=re.IGNORECASE)
        if len(parts) != 2:
            errors.append(
                f"Invalid range for {field}: '{normalized}' - use [start TO end]"
            )
            return

        start_raw, end_raw = parts[0].strip(), parts[1].strip()
        if not self._is_valid_date_endpoint(field, start_raw):
            errors.append(f"Invalid range start for {field}: '{start_raw}'")
        if not self._is_valid_date_endpoint(field, end_raw):
            errors.append(f"Invalid range end for {field}: '{end_raw}'")

        if start_raw == "*" or end_raw == "*":
            return

        start_key = self._date_sort_key(start_raw)
        end_key = self._date_sort_key(end_raw)
        if start_key and end_key and start_key > end_key:
            errors.append(
                f"Invalid range for {field}: start '{start_raw}' is after end '{end_raw}'"
            )

    def _is_valid_date_endpoint(self, field: str, raw: str) -> bool:
        if raw == "*":
            return True
        if field in YEAR_ONLY_DATE_FIELDS:
            return bool(YEAR_PATTERN.fullmatch(raw))
        return bool(YEAR_PATTERN.fullmatch(raw) or self._is_valid_date(raw))

    def _date_sort_key(self, raw: str) -> Optional[Tuple[int, int, int]]:
        if YEAR_PATTERN.fullmatch(raw):
            return int(raw), 1, 1
        if DATE_PATTERN.fullmatch(raw):
            try:
                parsed = datetime.strptime(raw, "%Y-%m-%d")
                return parsed.year, parsed.month, parsed.day
            except ValueError:
                return None
        return None

    def _validate_field_specific_format(
        self, field: str, normalized: str, errors: List[str]
    ) -> None:
        single_value_identifier_fields = {"PMCID", "ISSN", "ESSN", "ISBN", "DOI"}
        if field in single_value_identifier_fields and "," in normalized:
            errors.append(
                f"Multiple comma-separated values are not allowed for field '{field}'"
            )
            return

        if field == "AUTHORID" and not ORCID_PATTERN.fullmatch(normalized):
            errors.append(f"Invalid AUTHORID format: '{normalized}'")
        elif field in {"CITES", "REFFED_BY"}:
            match = CITATION_LINK_PATTERN.fullmatch(normalized)
            if not match:
                errors.append(
                    f"Invalid {field} format: '{normalized}' - expected identifier_source"
                )
            else:
                source = match.group(1).upper()
                if source not in FIXED_VALUE_FIELDS["SRC"]:
                    errors.append(f"Invalid source suffix in {field}: '{normalized}'")
        elif field == "PMCID" and not PMCID_PATTERN.fullmatch(normalized):
            errors.append(f"Invalid PMCID format: '{normalized}'")
        elif field in {"ISSN", "ESSN"} and not ISSN_PATTERN.fullmatch(normalized):
            errors.append(f"Invalid {field} format: '{normalized}'")
        elif field == "ISBN" and not self._is_valid_isbn(normalized):
            errors.append(f"Invalid ISBN format: '{normalized}'")
        elif field == "DOI" and not DOI_PATTERN.fullmatch(normalized):
            errors.append(f"Invalid DOI format: '{normalized}'")
        elif field == "BOOK_ID" and not BOOK_ID_PATTERN.fullmatch(normalized):
            errors.append(f"Invalid BOOK_ID format: '{normalized}'")
        elif field == "AUTH_MAN_ID" and not AUTH_MAN_ID_PATTERN.fullmatch(normalized):
            errors.append(f"Invalid AUTH_MAN_ID format: '{normalized}'")

    def _validate_field_specific_wildcards(
        self, field: str, normalized: str, value_kind: str, errors: List[str]
    ) -> None:
        has_wildcard = "*" in normalized or "?" in normalized
        if not has_wildcard:
            return

        if value_kind in {"quoted", "single_quoted"}:
            errors.append(
                f"Wildcards are not allowed inside quotes for field '{field}'"
            )
            return

        restricted_fields = {
            "AUTHORID",
            "CITES",
            "REFFED_BY",
            "ISSN",
            "ESSN",
            "ISBN",
            "DOI",
            "BOOK_ID",
            "AUTH_MAN_ID",
        }
        if field in restricted_fields:
            errors.append(f"Wildcards are not allowed for field '{field}'")
            return

        if field in DATE_FIELDS and "?" in normalized:
            errors.append(
                f"Question-mark wildcards are not allowed for field '{field}'"
            )
            return

        if field in DATE_FIELDS and value_kind != "range":
            errors.append(
                f"Wildcards are only allowed in date ranges for field '{field}'"
            )
            return

        if field in PURE_WILDCARD_RESTRICTED_FIELDS and re.fullmatch(
            r"[*?]+", normalized
        ):
            errors.append(
                f"Wildcard-only presence checks are not allowed for field '{field}' - use ANNOTATION_TYPE or HAS_TM when you mean annotation presence"
            )

    def _normalize_value(self, raw: str) -> str:
        raw = raw.strip()
        if (raw.startswith('"') and raw.endswith('"')) or (
            raw.startswith("'") and raw.endswith("'")
        ):
            return raw[1:-1].strip()
        return raw

    def _contains_unquoted_whitespace(self, normalized: str) -> bool:
        return bool(re.search(r"\s+", normalized))

    def _canonicalize_license(self, raw: str) -> str:
        normalized = raw.strip().upper().replace("_", "-")
        normalized = re.sub(r"\s+", "-", normalized)
        normalized = re.sub(r"-{2,}", "-", normalized)
        if normalized == "BY-NC-SA":
            return "CC-BY-NC-SA"
        return normalized

    def _normalize_author_role(self, raw: str) -> str:
        normalized = raw.strip().replace("–", "-").replace("—", "-")
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.casefold()

    def _is_valid_date(self, raw: str) -> bool:
        if not DATE_PATTERN.fullmatch(raw):
            return False
        try:
            datetime.strptime(raw, "%Y-%m-%d")
            return True
        except ValueError:
            return False

    def _is_valid_isbn(self, raw: str) -> bool:
        normalized = raw.replace("-", "").replace(" ", "")
        if len(normalized) == 10:
            return bool(re.fullmatch(r"\d{9}[\dXx]", normalized))
        if len(normalized) == 13:
            return normalized.isdigit()
        return False

    def _dedupe(self, values: List[str]) -> List[str]:
        return list(dict.fromkeys(values))
