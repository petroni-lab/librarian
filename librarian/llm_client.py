"""Minimal LLM client for a single OpenAI-compatible endpoint.

Point it at any OpenAI-compatible ``/chat/completions`` server (a self-hosted
vLLM instance, an OpenAI-compatible gateway, OpenAI itself, ...) with these
env vars — nothing else to configure:

    LLM_BASE_URL         endpoint, e.g. http://localhost:8000/v1   (the default)
    LLM_MODEL            model name to request
    LLM_API_KEY          bearer token (defaults to "EMPTY" for keyless servers)
    LLM_REASONING_EFFORT optional ``reasoning_effort`` value (e.g. "low") for
                         reasoning models (GLM-5.3+, etc). Servers that don't
                         recognize the field reject it once with a 400; the
                         client drops it and retries automatically.
"""

import json
import os
import random
import re
import time
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


class LLMClient:
    """A client for a single OpenAI-compatible chat-completions endpoint."""

    MAX_RETRIES = 6  # retry attempts for rate-limit / transient errors

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.base_url = (
            base_url or os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
        ).rstrip("/")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "EMPTY")
        self.model_name = model_name or os.getenv("LLM_MODEL")
        # Reasoning-model effort level (e.g. "low"); unset means don't send it.
        self.reasoning_effort = os.getenv("LLM_REASONING_EFFORT")

        # Optional sampling parameters this endpoint has rejected with a 400.
        self._unsupported: set = set()

        effort_note = (
            f" reasoning_effort={self.reasoning_effort}"
            if self.reasoning_effort
            else ""
        )
        print(f"LLMClient -> url={self.base_url} model={self.model_name}{effort_note}")

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
        timeout: int = 180,
    ) -> str:
        """Send one chat completion and return the assistant's text.

        Retries rate-limit (429), 5xx, and transient network errors with
        exponential backoff + jitter; 4xx client errors are not retried.

        One exception: a 400 that names an optional sampling parameter (``error
        .param``) makes us drop that parameter and retry immediately.
        """
        optional = {"temperature": temperature}
        if max_tokens is not None:
            optional["max_tokens"] = int(max_tokens)
        if self.reasoning_effort:
            optional["reasoning_effort"] = self.reasoning_effort

        last_exception = None
        for attempt in range(self.MAX_RETRIES):
            payload = {
                "model": self.model_name,
                "messages": messages,
                **{k: v for k, v in optional.items() if k not in self._unsupported},
            }
            try:
                response = requests.post(
                    self._url(), headers=self._headers(), json=payload, timeout=timeout
                )
                if response.status_code == 400:
                    last_exception = Exception(f"400: {response.text[:200]}")
                    if self._drop_rejected_param(response, optional):
                        continue  # same request, minus the rejected parameter
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

    def _drop_rejected_param(self, response: Any, optional: dict) -> bool:
        """Given a 400 code, mark the parameter the endpoint complained about as
        unsupported. Returns True if something was dropped and the request is
        worth retrying, False to let the 400 surface to the caller."""
        try:
            error = (response.json() or {}).get("error") or {}
        except ValueError:
            return False

        message = str(error.get("message") or "")
        param = error.get("param")
        if param not in optional:
            param = next((k for k in optional if f"'{k}'" in message), None)

        if param is None or param in self._unsupported:
            return False

        self._unsupported.add(param)
        print(
            f"LLMClient: {self.model_name} rejected '{param}' "
            f"({message[:120]}); retrying without it"
        )
        return True

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


def create_llm_client(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> LLMClient:
    """Build the LLM client. Kept as a factory so the agent's call site stays
    unchanged even if construction ever grows more logic."""
    return LLMClient(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
    )
