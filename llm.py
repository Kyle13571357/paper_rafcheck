#!/usr/bin/env python3
"""Model access layer.

One seam for every model call in the pipeline. Module H records the model as
an experimental variable ("model version + date + prompt + temperature"), so
the backend has to be swappable without touching ask.py, check.py or
vision_verify.py.

Design: an abstract `LLMProvider` defines the contract; each backend
subclasses it. DeepSeek and OpenAI differ only in base URL and model ids, so
they share an OpenAI-compatible base class; Ollama speaks a different wire
format and overrides the request/response methods. Callers resolve a provider
through `get_provider()` and never branch on which one they got.

    REFCHECK_PROVIDER      deepseek (default) | openai | ollama
    DEEPSEEK_API_KEY       required for the deepseek provider
    REFCHECK_TEXT_MODEL    override the text model id
    REFCHECK_VISION_MODEL  override the vision model id

Vision note: DeepSeek serves no vision model, so its `complete_vision` raises
rather than silently dropping the image. An image that never arrives is the
exact failure vision_verify.py's token check exists to catch, and a provider
that quietly degrades to text would defeat it.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}
TRANSPORT_ERRORS = (urllib.error.URLError, TimeoutError, OSError)


class LLMError(RuntimeError):
    """Any failure originating in the model layer."""


class LLMProvider(ABC):
    """Contract every backend implements."""

    name: str = "abstract"
    default_text_model: str = ""
    default_vision_model: str | None = None
    key_env: str | None = None
    base_url: str = ""

    def __init__(self) -> None:
        self.text_model = os.environ.get("REFCHECK_TEXT_MODEL") or self.default_text_model
        self.vision_model = os.environ.get("REFCHECK_VISION_MODEL") or self.default_vision_model
        self.api_key = os.environ.get(self.key_env) if self.key_env else None

    # -- capability / readiness -------------------------------------------
    @property
    def supports_vision(self) -> bool:
        return bool(self.vision_model)

    @property
    def is_ready(self) -> bool:
        return not self.key_env or bool(self.api_key)

    def require_ready(self) -> None:
        if not self.is_ready:
            raise LLMError(
                f"{self.key_env} is not set, so the {self.name} provider cannot be "
                f"used. Export it, or select another backend with "
                f"REFCHECK_PROVIDER=openai|ollama."
            )

    def describe(self) -> str:
        """Provenance string recorded alongside every experimental result."""
        return f"{self.name}:{self.text_model}"

    def info(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "base_url": self.base_url,
            "text_model": self.text_model,
            "vision_model": self.vision_model,
            "supports_vision": self.supports_vision,
            "ready": self.is_ready,
            "key_env": self.key_env,
        }

    # -- subclasses supply the wire format --------------------------------
    @abstractmethod
    def _endpoint(self) -> str: ...

    @abstractmethod
    def _headers(self) -> dict[str, str]: ...

    @abstractmethod
    def _text_payload(self, prompt: str, json_mode: bool, temperature: float,
                      max_tokens: int) -> dict[str, Any]: ...

    @abstractmethod
    def _vision_payload(self, prompt: str, image_b64: str, json_mode: bool,
                        temperature: float, max_tokens: int) -> dict[str, Any]: ...

    @abstractmethod
    def _extract(self, response: dict[str, Any]) -> str: ...

    # -- shared transport --------------------------------------------------
    def _post(self, payload: dict[str, Any], timeout: int, attempts: int = 3) -> str:
        body = json.dumps(payload).encode()
        last: Exception | None = None
        for attempt in range(attempts):
            try:
                req = urllib.request.Request(self._endpoint(), data=body,
                                             headers=self._headers())
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    return self._extract(json.load(r))
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode()[:400]
                except Exception:
                    pass
                # a 4xx that is not rate limiting will not fix itself
                if e.code not in RETRYABLE_STATUS:
                    raise LLMError(f"{e} {detail}") from e
                last = e
            except TRANSPORT_ERRORS as e:
                last = e
            if attempt < attempts - 1:
                time.sleep(5 * (attempt + 1))
        raise LLMError(str(last))

    def complete(self, prompt: str, json_mode: bool = True, temperature: float = 0.0,
                 timeout: int = 300, max_tokens: int = 2048) -> str:
        self.require_ready()
        return self._post(self._text_payload(prompt, json_mode, temperature, max_tokens),
                          timeout)

    def complete_vision(self, prompt: str, png_bytes: bytes, json_mode: bool = True,
                        temperature: float = 0.0, timeout: int = 600,
                        max_tokens: int = 2048) -> str:
        if not self.supports_vision:
            raise LLMError(
                f"provider {self.name!r} has no vision model, so the page image "
                f"cannot be sent. Set REFCHECK_PROVIDER to a vision-capable "
                f"backend (openai, or ollama for local) before running "
                f"vision_verify.py."
            )
        self.require_ready()
        b64 = base64.b64encode(png_bytes).decode()
        return self._post(
            self._vision_payload(prompt, b64, json_mode, temperature, max_tokens),
            timeout)


class OpenAICompatibleProvider(LLMProvider):
    """Shared by every backend speaking the OpenAI chat-completions shape."""

    def _endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"}

    def _text_payload(self, prompt, json_mode, temperature, max_tokens):
        payload = {"model": self.text_model,
                   "messages": [{"role": "user", "content": prompt}],
                   "temperature": temperature, "max_tokens": max_tokens}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _vision_payload(self, prompt, image_b64, json_mode, temperature, max_tokens):
        payload = {
            "model": self.vision_model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ]}],
            "temperature": temperature, "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _extract(self, response):
        return response["choices"][0]["message"]["content"]


class DeepSeekProvider(OpenAICompatibleProvider):
    name = "deepseek"
    base_url = "https://api.deepseek.com"
    key_env = "DEEPSEEK_API_KEY"
    default_text_model = "deepseek-chat"
    default_vision_model = None          # DeepSeek serves no vision model


class OpenAIProvider(OpenAICompatibleProvider):
    name = "openai"
    base_url = "https://api.openai.com/v1"
    key_env = "OPENAI_API_KEY"
    default_text_model = "gpt-4o"
    default_vision_model = "gpt-4o"


class OllamaProvider(LLMProvider):
    """Local runtime. Kept selectable but never the default: 7B inference was
    measured at 200-2200s per vision page on the development machine."""

    name = "ollama"
    base_url = "http://localhost:11434"
    key_env = None
    default_text_model = "qwen2.5:7b"
    default_vision_model = "qwen2.5vl:7b"

    def _endpoint(self):
        return f"{self.base_url}/api/generate"

    def _headers(self):
        return {"Content-Type": "application/json"}

    def _text_payload(self, prompt, json_mode, temperature, max_tokens):
        payload = {"model": self.text_model, "prompt": prompt, "stream": False,
                   "options": {"temperature": temperature}}
        if json_mode:
            payload["format"] = "json"
        return payload

    def _vision_payload(self, prompt, image_b64, json_mode, temperature, max_tokens):
        payload = {"model": self.vision_model, "prompt": prompt,
                   "images": [image_b64], "stream": False,
                   "options": {"temperature": temperature}}
        if json_mode:
            payload["format"] = "json"
        return payload

    def _extract(self, response):
        return response.get("response", "")


PROVIDERS: dict[str, type[LLMProvider]] = {
    "deepseek": DeepSeekProvider,
    "openai": OpenAIProvider,
    "ollama": OllamaProvider,
}


def provider_name() -> str:
    return os.environ.get("REFCHECK_PROVIDER", "deepseek").strip().lower()


def get_provider(name: str | None = None) -> LLMProvider:
    """Factory. Reads the environment each call so tests and the API service
    can switch backend without reimporting the module."""
    key = (name or provider_name()).strip().lower()
    if key not in PROVIDERS:
        raise LLMError(f"unknown provider {key!r}; expected one of {sorted(PROVIDERS)}")
    return PROVIDERS[key]()


# --------------------------------------------------------------------------
# Module-level helpers: the CLI entry points call these, so they keep working
# unchanged while the object model underneath does the real work.
# --------------------------------------------------------------------------

def complete(prompt: str, **kw) -> str:
    return get_provider().complete(prompt, **kw)


def complete_vision(prompt: str, png_bytes: bytes, **kw) -> str:
    return get_provider().complete_vision(prompt, png_bytes, **kw)


def describe() -> str:
    return get_provider().describe()


def config() -> dict[str, Any]:
    return get_provider().info()


def require_key(cfg_or_provider: Any = None) -> None:
    get_provider().require_ready()


def parse_json_reply(text: str) -> dict[str, Any] | None:
    """Models wrap JSON in prose or fences often enough that this belongs in
    one place. Returns None rather than raising, so callers can record the
    unparseable output instead of losing the whole run."""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) > 1:
            text = parts[1]
            if text.lstrip().lower().startswith("json"):
                text = text.lstrip()[4:]
            text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                return None
    return None


if __name__ == "__main__":
    p = get_provider()
    for k, v in p.info().items():
        print(f"{k:16s}: {v}")
