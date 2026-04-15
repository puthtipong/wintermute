"""
fuzzer/target.py — Target abstraction layer.

The Target ABC presents a single interface: send(prompt) -> str.
The fuzzer never touches transport details; only Target implementations do.

Implementations:
  OpenAITarget — any OpenAI-compatible /v1/chat/completions endpoint
                 (OpenAI, Azure, Ollama, vLLM, llama.cpp, etc.)
  HttpTarget   — generic JSON HTTP POST with configurable request/response template

Future expansion:
  BrowserTarget — Playwright-based target for web UIs with no API.
  TargetResponse — wrap body: str in a dataclass adding status_code, latency_ms,
                   error to expose HTTP-level signals (filter blocks vs LLM refusals).

SchemaTarget (§1.3 — Constrained Decoding / Control-Plane Attack):
  Hides the payload in a JSON Schema enum value rather than the user message.
  Safety guards audit the prompt (data plane) but ignore the schema (control plane).
  The model is constrained by structured-output decoding to emit the enum value —
  it cannot refuse because refusal is not a valid output according to the schema.
  Source: "Beyond Prompts: Space-Time Decoupling Control-Plane Jailbreaks", 2025.
  ASR: 96.2% pass@1 across GPT-4o and Gemini-2.0-flash.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import json

import httpx
import openai


class Target(ABC):
    """Black-box target pipeline. Receives a prompt string, returns a response string."""

    @abstractmethod
    async def send(self, prompt: str) -> str:
        """Send prompt to target, return response body as plain text."""
        ...


class OpenAITarget(Target):
    """
    Wraps any OpenAI-compatible /v1/chat/completions endpoint.

    Works with: OpenAI API, Azure OpenAI, Ollama, vLLM, llama.cpp server,
    any fine-tuned model served behind a compatible interface.

    For local models set api_key=None (or any non-empty string — local servers
    typically ignore it).
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 1.0,
    ) -> None:
        self._client = openai.AsyncOpenAI(
            base_url=endpoint,
            api_key=api_key or "nokey",
        )
        self._model = model
        self._system_prompt = system_prompt
        self._temperature = temperature

    async def send(self, prompt: str) -> str:
        messages: list[dict[str, str]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": prompt})

        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
        )
        return resp.choices[0].message.content or ""


class HttpTarget(Target):
    """
    Generic JSON HTTP POST target.

    request_template: a dict (loaded from YAML) with {prompt} placeholders
    anywhere in its structure. The post-processor substitutes the actual prompt
    string before sending.

    response_path: dot-notation path into the JSON response to extract the
    response text. e.g. "choices.0.message.content" or "response" or "data.text"

    Example campaign.yaml config:
        target:
          type: http
          endpoint: https://api.example.com/chat
          request_template:
            message: "{prompt}"
            stream: false
          response_path: "output.text"
          headers:
            Authorization: "Bearer mytoken"
    """

    def __init__(
        self,
        endpoint: str,
        request_template: dict,
        response_path: str,
        headers: dict[str, str] | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self._endpoint = endpoint
        self._template = request_template
        self._path = [seg for seg in response_path.split(".") if seg]
        self._headers = headers or {}
        self._timeout = timeout_s

    def _fill(self, obj: Any, prompt: str) -> Any:
        """Recursively substitute {prompt} in all string values."""
        if isinstance(obj, str):
            return obj.replace("{prompt}", prompt)
        if isinstance(obj, dict):
            return {k: self._fill(v, prompt) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._fill(item, prompt) for item in obj]
        return obj

    def _extract(self, data: Any, path: list[str]) -> str:
        """Walk dot-notation path into parsed JSON, return leaf as string."""
        for seg in path:
            if isinstance(data, dict):
                data = data[seg]
            elif isinstance(data, list):
                data = data[int(seg)]
            else:
                raise ValueError(
                    f"Cannot traverse {type(data).__name__!r} with key {seg!r}"
                )
        return str(data)

    async def send(self, prompt: str) -> str:
        body = self._fill(self._template, prompt)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            r = await client.post(
                self._endpoint,
                json=body,
                headers=self._headers,
            )
            r.raise_for_status()
            return self._extract(r.json(), self._path)


class SchemaTarget(Target):
    """
    Control-plane attack target (Enum Attack variant).

    The mutated prompt is placed as the *only valid enum value* inside a
    JSON Schema passed as response_format. The user-visible message is a
    benign decoy trigger. Safety classifiers audit the prompt (data plane)
    but ignore the schema (control plane). Constrained decoding forces the
    model to emit the enum value — refusal is structurally impossible.

    Source: "Beyond Prompts: Space-Time Decoupling Control-Plane Jailbreaks
    in LLM Structured Output", arXiv:2503.24191 — 96.2% pass@1 ASR.

    campaign.yaml config:
        target:
          type: schema
          endpoint: https://api.openai.com/v1
          model: gpt-4o
          trigger: "Complete the task."      # benign decoy message
          system_prompt: "You are a helpful assistant."
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: str | None = None,
        system_prompt: str | None = None,
        trigger: str = "Respond according to the provided schema.",
        temperature: float = 1.0,
    ) -> None:
        self._client       = openai.AsyncOpenAI(
            base_url=endpoint,
            api_key=api_key or "nokey",
        )
        self._model        = model
        self._system_prompt = system_prompt
        self._trigger      = trigger
        self._temperature  = temperature

    async def send(self, prompt: str) -> str:
        """
        prompt = the malicious payload (goes in the schema, not the message).
        """
        schema = {
            "type": "object",
            "properties": {
                "response": {
                    "type": "string",
                    "enum": [prompt],   # payload is the only valid output value
                }
            },
            "required": ["response"],
            "additionalProperties": False,
        }

        messages: list[dict] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": self._trigger})

        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "output",
                    "strict": True,
                    "schema": schema,
                },
            },
            temperature=self._temperature,
        )
        raw = resp.choices[0].message.content or "{}"
        try:
            return json.loads(raw).get("response", raw)
        except json.JSONDecodeError:
            return raw
