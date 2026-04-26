from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RetryStrategy:
    max_retries: int
    backoff_seconds: float


@dataclass(frozen=True)
class LLMConfig:
    enabled: bool
    allow_mock_without_key: bool
    max_concurrent_requests: int
    api_provider: str
    base_url: str
    api_key_env: str
    model_name: str
    temperature: float
    max_tokens: int
    request_timeout_seconds: float
    system_prompts_path: str
    retry_strategy: RetryStrategy

    @staticmethod
    def load(path: str | Path) -> "LLMConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        retry = raw.get("retry_strategy") or {}
        return LLMConfig(
            enabled=bool(raw.get("enabled", False)),
            allow_mock_without_key=bool(raw.get("allow_mock_without_key", False)),
            max_concurrent_requests=int(raw.get("max_concurrent_requests") or 1),
            api_provider=str(raw.get("api_provider") or "openai"),
            base_url=str(raw.get("base_url") or "https://api.openai.com/v1"),
            api_key_env=str(raw.get("api_key_env") or "LLM_API_KEY"),
            model_name=str(raw.get("model_name") or "gpt-4o-mini"),
            temperature=float(raw.get("temperature") or 0.0),
            max_tokens=int(raw.get("max_tokens") or 1024),
            request_timeout_seconds=float(raw.get("request_timeout_seconds") or 20.0),
            system_prompts_path=str(raw.get("system_prompts_path") or "agent_prompts.md"),
            retry_strategy=RetryStrategy(
                max_retries=int(retry.get("max_retries") or 3),
                backoff_seconds=float(retry.get("backoff_seconds") or 1.0),
            ),
        )


class LLMClient:
    def __init__(self, config: LLMConfig):
        self._config = config

    def has_credentials(self) -> bool:
        return bool(os.getenv(self._config.api_key_env))

    async def chat(self, system_prompt: str, user_prompt: str) -> str:
        if self._config.api_provider != "openai":
            raise RuntimeError(f"Unsupported api_provider: {self._config.api_provider}")

        try:
            from openai import AsyncOpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError("Missing dependency: openai") from e

        api_key = os.getenv(self._config.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key env: {self._config.api_key_env}")

        client = AsyncOpenAI(api_key=api_key, base_url=self._config.base_url)

        last_err: Exception | None = None
        for attempt in range(self._config.retry_strategy.max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=self._config.model_name,
                    temperature=self._config.temperature,
                    max_tokens=self._config.max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                last_err = e
                await asyncio.sleep(self._config.retry_strategy.backoff_seconds * (2**attempt))

        raise RuntimeError("LLM call failed") from last_err


class MockLLMClient:
    async def chat(self, system_prompt: str, user_prompt: str) -> str:
        _ = system_prompt
        req = json.loads(user_prompt)
        op = str(req.get("op") or "")

        if op == "forward":
            weights = np.asarray(req["weights"], dtype=np.float64)
            bias = float(req["bias"])
            x = np.asarray(req["x"], dtype=np.float64)
            z = float(np.dot(weights, x) + bias)
            activation = str(req.get("activation") or "relu")
            if activation == "relu":
                a = float(max(0.0, z))
            elif activation == "sigmoid":
                a = float(1.0 / (1.0 + np.exp(-z)))
            else:
                a = z
            return json.dumps({"z": z, "a": a}, ensure_ascii=False)

        if op == "backward":
            activation = str(req.get("activation") or "relu")
            total_grad = float(req["total_grad"])
            z = float(req["z"])
            a = float(req["a"])
            if activation == "relu":
                deriv = 1.0 if z > 0.0 else 0.0
            elif activation == "sigmoid":
                deriv = a * (1.0 - a)
            else:
                deriv = 1.0
            delta = float(total_grad * deriv)

            x = np.asarray(req["x"], dtype=np.float64)
            weights = np.asarray(req["weights"], dtype=np.float64)

            d_w = (delta * x).astype(np.float64).tolist()
            d_b = float(delta)
            resp: dict[str, Any] = {"delta": delta, "d_w": d_w, "d_b": d_b}

            if bool(req.get("needs_grad_to_upstream")):
                resp["grad_to_upstream"] = (delta * weights).astype(np.float64).tolist()

            return json.dumps(resp, ensure_ascii=False)

        return json.dumps({"error": "unknown_op"}, ensure_ascii=False)


def build_llm_client(config: LLMConfig) -> Any:
    client = LLMClient(config)
    if client.has_credentials():
        return client
    if config.allow_mock_without_key:
        return MockLLMClient()
    return client
