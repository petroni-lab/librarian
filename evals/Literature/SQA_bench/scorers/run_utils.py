import json
import jsonlines
import logging
import time
from typing import Any, Dict, Optional
import os
import litellm

LOGGER = logging.getLogger(__name__)
os.environ["LITELLM_LOG"] = "DEBUG"


def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
    json_start = response.find("{")
    json_end = response.rfind("}") + 1
    if json_start == -1 or json_end == -1:
        return None

    try:
        return json.loads(response[json_start:json_end])
    except json.JSONDecodeError:
        try:
            return json.loads(response[json_start:json_end] + "]}")
        except json.JSONDecodeError:
            LOGGER.warning(
                f"Could not decode JSON from response: {response[json_start:json_end]}"
            )
        return None


def run_chatopenai(
    model_name: str,
    system_prompt: Optional[str],
    user_prompt: str,
    json_mode: bool = False,
    **chat_kwargs,
) -> str:
    chat_kwargs["temperature"] = chat_kwargs.get("temperature", 0)
    if json_mode:
        chat_kwargs["response_format"] = {"type": "json_object"}
    msgs = (
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if system_prompt is not None
        else [{"role": "user", "content": user_prompt}]
    )
    max_retries = 8
    for attempt in range(max_retries):
        try:
            resp = litellm.completion(
                model=model_name,
                messages=msgs,
                **chat_kwargs,
            )
            return resp.choices[0].message.content
        except litellm.exceptions.RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            wait = 2**attempt  # 1s, 2s, 4s, 8s, 16s, 32s, 64s
            LOGGER.warning(
                f"Rate limit hit, retrying in {wait}s (attempt {attempt + 1}/{max_retries}): {e}"
            )
            time.sleep(wait)


def load_jsonlines(file):
    with jsonlines.open(file, "r") as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst


def save_file_jsonl(data, fp):
    with jsonlines.open(fp, mode="w") as writer:
        writer.write_all(data)
