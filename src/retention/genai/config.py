"""Validated, provider-neutral generation settings and policy catalogue."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from retention.utils.config import load_config


class FallbackConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True


class GenAIConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: str
    model: str
    temperature: float = Field(ge=0, le=2)
    request_timeout: int = Field(ge=1, le=120)
    max_retries: int = Field(ge=0, le=5)
    max_output_tokens: int = Field(ge=128, le=8192)
    prompt_version: str
    policy_version: str
    fallback: FallbackConfig


def load_genai_config(path: str | Path = "configs/genai.yaml") -> GenAIConfig:
    return GenAIConfig.model_validate(load_config(path))


def load_policies(path: str | Path = "configs/retention_policies.yaml") -> dict:
    value = load_config(path)
    if not isinstance(value.get("actions"), dict) or not value.get("version"):
        raise ValueError("Invalid retention policy catalogue")
    return value
