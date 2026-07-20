"""Minimal OpenAI-compatible LLM client.

Talks to any OpenAI-compatible ``/chat/completions`` endpoint — a self-hosted
vLLM server or a remote provider. The full agent's client also did Phoenix
tracing, Vertex AI auth, and primary/fallback failover; none of that is needed
to run the librarian, so this version keeps only what the flow uses:
``chat_completion``, ``batch_chat_completion``, and ``is_external_api``.

Backend selection (all optional, via env vars):
  LLM_BASE_URL   endpoint, e.g. http://localhost:8000/v1 (the default)
  LLM_MODEL      model name to request
  LLM_API_KEY    bearer token (defaults to "EMPTY" for keyless vLLM)
  LLM_PROVIDER   "vllm" (default) or "openai"/"zai"/... — only affects the
                 default base_url and the thinking-flag wire format.
"""

import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional

import requests


def strip_code_fences(text: str) -> str:
    """Strip a ```json ... ``` (or bare ```...```) fence an LLM wrapped its response in."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def parse_json_response(text: str) -> Any:
    """Parse JSON from an LLM response: try the whole (fence-stripped) string
    first, then fall back to the first {...}/[...] span. Returns None on failure."""
    cleaned = strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        pass
    match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


# Thread pool used by batch_chat_completion to fan out concurrent requests.
_BATCH_WORKERS = 8


class LLMClient:
    """A general-purpose client for OpenAI-compatible LLM APIs."""

    MAX_RETRIES = 6  # retry attempts for rate-limit / transient errors
    _EXTERNAL_MIN_INTERVAL = 1.0  # min seconds between remote-provider requests

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        thinking: bool = False,
    ):
        self.thinking = thinking

        provider = os.getenv("LLM_PROVIDER", "vllm").split(":", 1)[0].strip().lower()

        # Base URL: explicit arg > LLM_BASE_URL > provider default.
        default_urls = {
            "openai": "https://api.openai.com/v1",
            "zai": "https://api.z.ai/api/paas/v4",
            "vllm": "http://localhost:8000/v1",
        }
        resolved = (
            base_url
            or os.getenv("LLM_BASE_URL")
            or default_urls.get(provider, default_urls["vllm"])
        )
        self.base_url = resolved.rstrip("/")

        url_lower = self.base_url.lower()
        self._is_openai = "api.openai.com" in url_lower
        self._is_zai = "api.z.ai" in url_lower
        # A "vLLM-like" backend is any self-hosted OpenAI-compatible server: it
        # accepts the legacy max_tokens field and chat_template_kwargs thinking flag.
        self._is_vllm_backend = not self._is_openai and not self._is_zai
        self._is_external_api = self._is_openai or self._is_zai

        self.api_key = api_key or os.getenv("LLM_API_KEY", "EMPTY")
        self.model_name = model_name or os.getenv("LLM_MODEL")

        print(
            f"LLMClient -> url={self.base_url} model={self.model_name} "
            f"thinking={'ON' if thinking else 'OFF'}"
        )
        self._last_request_time = 0.0

    @property
    def is_external_api(self) -> bool:
        """True for a remote provider (OpenAI/z.ai), False for self-hosted vLLM."""
        return self._is_external_api

    def _url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    @staticmethod
    def _extract_content(choice: dict) -> str:
        """Pull the assistant text out of a choice, tolerating content-block formats."""
        message = choice.get("message") or {}
        content = message.get("content", "")
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    parts.append(str(part.get("text") or part.get("content") or ""))
            content = "\n".join(p for p in parts if p)
        elif content is None:
            content = ""
        if not content:
            # Some reasoning backends put the answer in a different field.
            content = (
                message.get("reasoning_content")
                or message.get("output_text")
                or choice.get("text")
                or ""
            )
        return str(content)

    def chat_completion(
        self,
        messages: List[dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        thinking: Optional[bool] = None,
        timeout: int = 180,
    ) -> str:
        """Send one chat completion and return the assistant's text.

        Retries rate-limit (429), 5xx, and transient network errors with
        exponential backoff + jitter; 4xx client errors are not retried.
        """
        use_thinking = thinking if thinking is not None else self.thinking

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
        }
        # OpenAI reasoning models want max_completion_tokens; vLLM/z.ai take max_tokens.
        tokens_key = "max_tokens" if not self._is_openai else "max_completion_tokens"
        if max_tokens is not None:
            payload[tokens_key] = max(1, int(max_tokens))
        if self._is_zai:
            payload["thinking"] = {"type": "enabled" if use_thinking else "disabled"}
        elif self._is_vllm_backend:
            payload["chat_template_kwargs"] = {"enable_thinking": use_thinking}

        last_exception = None
        for attempt in range(self.MAX_RETRIES):
            # Throttle remote providers to stay under RPM limits.
            if self._is_external_api:
                elapsed = time.time() - self._last_request_time
                if elapsed < self._EXTERNAL_MIN_INTERVAL:
                    time.sleep(self._EXTERNAL_MIN_INTERVAL - elapsed)

            try:
                self._last_request_time = time.time()
                response = requests.post(
                    self._url(), headers=self._headers(), json=payload, timeout=timeout
                )
                if response.status_code == 429:
                    last_exception = Exception(f"429: {response.text[:200]}")
                    time.sleep(
                        self._backoff(attempt, response.headers.get("Retry-After"))
                    )
                    continue
                response.raise_for_status()
                choice = (response.json().get("choices") or [{}])[0]
                return self._extract_content(choice)

            except requests.exceptions.RequestException as exc:
                last_exception = exc
                status = getattr(getattr(exc, "response", None), "status_code", None)
                transient = (status is not None and status >= 500) or isinstance(
                    exc,
                    (
                        requests.exceptions.ConnectionError,
                        requests.exceptions.Timeout,
                        requests.exceptions.ChunkedEncodingError,
                    ),
                )
                if attempt < self.MAX_RETRIES - 1 and transient:
                    time.sleep(self._backoff(attempt))
                    continue
                raise Exception(f"LLM API request failed: {exc}") from exc

        raise Exception(
            f"LLM API request failed after {self.MAX_RETRIES} retries: {last_exception}"
        )

    @staticmethod
    def _backoff(attempt: int, retry_after: Optional[str] = None) -> float:
        """Seconds to wait before the next retry: honor Retry-After, else 2**attempt
        capped at 120s, with +/-25% jitter to avoid a thundering herd."""
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
        wait = min(2**attempt, 120)
        return max(1.0, wait + wait * 0.25 * (2 * random.random() - 1))

    def batch_chat_completion(
        self,
        list_of_message_lists: List[List[dict]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        thinking: Optional[bool] = None,
    ) -> List[str]:
        """Run N independent conversations concurrently, returning results in order.

        The full agent used vLLM's native /chat/completions/batch route; here we
        simply fan the calls out over a thread pool, which works against any
        OpenAI-compatible endpoint and keeps the code readable.
        """
        if not list_of_message_lists:
            return []
        with ThreadPoolExecutor(max_workers=_BATCH_WORKERS) as pool:
            futures = [
                pool.submit(
                    self.chat_completion,
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    thinking=thinking,
                )
                for messages in list_of_message_lists
            ]
            return [f.result() for f in futures]  # submission order == input order


def create_llm_client(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    thinking: bool = False,
) -> LLMClient:
    """Build the LLM client. Kept as a factory so the agent's call site is
    unchanged from the full version (which selected between backends here)."""
    return LLMClient(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
        thinking=thinking,
    )
