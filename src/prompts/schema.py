"""Helpers for validating model outputs against JSON Schema."""
from __future__ import annotations

import json
from typing import Any, Dict

from jsonschema import Draft202012Validator, ValidationError


class SchemaValidationError(RuntimeError):
    """Raised when the model output fails JSON schema validation."""

    def __init__(self, message: str, errors: list[str]):
        super().__init__(message)
        self.errors = errors


def parse_schema(schema_str: str) -> Dict[str, Any]:
    """Parse a JSON schema string into a dictionary."""

    return json.loads(schema_str) if schema_str else {}


def validate_output(schema_str: str, candidate: Dict[str, Any]) -> None:
    """Validate `candidate` using the provided schema string.

    Raises SchemaValidationError when validation fails.
    """

    if not schema_str:
        return
    schema = parse_schema(schema_str)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(candidate), key=lambda e: e.path)
    if errors:
        messages = [f"{'/'.join(map(str, err.path)) or '<root>'}: {err.message}" for err in errors]
        raise SchemaValidationError("Response failed schema validation", messages)
