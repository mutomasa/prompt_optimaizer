"""Prompt loading and rendering utilities.

The optimizer relies on structured prompt templates (YAML/TOML) with knob
parameters. This module loads the template and renders it with a jinja2 context.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import toml
import yaml
from jinja2 import Template


@dataclass(frozen=True)
class PromptConfig:
    """Structured prompt definition loaded from disk."""

    task: str
    system: str
    user: str
    constraints: str
    knobs: Dict[str, list]
    defaults: Dict[str, Any]
    schema: str

    def merged_knobs(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Merge defaults with overrides while keeping known keys only."""

        data = {**self.defaults}
        for key, value in overrides.items():
            if key not in self.knobs:
                raise KeyError(f"Unknown knob: {key}")
            data[key] = value
        return data


def _read_prompt(path: Path) -> Dict[str, Any]:
    ext = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if ext in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    if ext == ".toml":
        return toml.loads(text)
    raise ValueError("Supported extensions are .yaml, .yml and .toml")


def load_prompt(path: str | Path) -> PromptConfig:
    """Load a prompt definition from disk into a PromptConfig."""

    p = Path(path)
    raw = _read_prompt(p)
    tpl = raw.get("template", {})
    return PromptConfig(
        task=raw["task"],
        system=tpl["system"],
        user=tpl["user"],
        constraints=tpl.get("constraints", ""),
        knobs=raw.get("knobs", {}),
        defaults=raw.get("defaults", {}),
        schema=raw.get("schema", ""),
    )


def render_prompt(cfg: PromptConfig, instruction: str, context: str, knobs: Dict[str, Any]) -> Dict[str, str]:
    """Render the prompt with the provided knobs and inputs."""

    data = cfg.merged_knobs(knobs)
    payload = {**data, "instruction": instruction, "context": context, "schema": cfg.schema}
    return {
        "system": Template(cfg.system).render(**payload),
        "user": Template(cfg.user).render(**payload),
        "constraints": Template(cfg.constraints).render(**payload),
    }


def list_knob_space(cfg: PromptConfig) -> Dict[str, list]:
    """Return the available options for each knob."""

    return cfg.knobs
