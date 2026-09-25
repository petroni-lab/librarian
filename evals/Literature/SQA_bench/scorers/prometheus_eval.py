import argparse
import json
import logging
import os
import re
import numpy as np
from urllib.parse import urlparse

from fastchat.conversation import get_conv_template
from transformers import LlamaForCausalLM

# from vllm import LLM, SamplingParams (imported lazily)
import torch
from collections import Counter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

prompt_template = (
    "An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given."
    "1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general."
    "2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric."
    '3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"'
    "4. Please do not generate any other opening, closing, and explanations."
    "###The instruction to evaluate:\n{instruction}\n"
    "###Response to evaluate:\n{response}\n"
    "###Reference Answer (Score 5):\n{reference_answer}"
    "###Score Rubrics:\n"
    "\n\n[{criteria_description}]"
    "\nScore 1: {score1_description}"
    "\nScore 2: {score2_description}"
    "\nScore 3: {score3_description}"
    "\nScore 4: {score4_description}"
    "\nScore 5: {score5_description}"
)

prompt_template_wo_header = (
    "An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given."
    "1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general."
    "2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric."
    '3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"'
    "4. Please do not generate any other opening, closing, and explanations."
    "\n{instruction}"
    "\n{response}"
    "\n{reference_answer}"
    "\n\n[{criteria_description}]"
    "\nScore 1: {score1_description}"
    "\nScore 2: {score2_description}"
    "\nScore 3: {score3_description}"
    "\nScore 4: {score4_description}"
    "\nScore 5: {score5_description}"
)


prompt_template_no_reference = (
    "An instruction (might include an Input inside it), a response to evaluate and a score rubric representing a evaluation criteria are given."
    "1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general."
    "2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric."
    '3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"'
    "4. Please do not generate any other opening, closing, and explanations."
    "\n{instruction}"
    "\n{response}"
    "\n\n[{criteria_description}]"
    "\nScore 1: {score1_description}"
    "\nScore 2: {score2_description}"
    "\nScore 3: {score3_description}"
    "\nScore 4: {score4_description}"
    "\nScore 5: {score5_description}"
)


def read_txt_file(file_path):
    """
    Read a text file to string
    """
    with open(file_path, "r") as file:
        return file.read()


def read_json(file_path):
    """
    Read a json file to dict
    """
    with open(file_path, "r") as file:
        return json.load(file)


def load_records_file(file_path):
    """Load records from a JSON list, JSONL file, or {"data": [...]} wrapper."""
    if file_path.lower().endswith(".jsonl"):
        records = []
        with open(file_path, "r") as infile:
            for line in infile:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    loaded = read_json(file_path)
    if isinstance(loaded, dict) and isinstance(loaded.get("data"), list):
        return loaded["data"]
    if isinstance(loaded, list):
        return loaded
    raise ValueError(f"Unsupported records file format: {file_path}")


def load_batch_results(batch_input_path):
    """
    Load response entries from a JSON file, JSONL file, or a directory containing one such file.
    """
    resolved_path = batch_input_path
    if os.path.isdir(batch_input_path):
        candidates = sorted(
            [
                os.path.join(batch_input_path, name)
                for name in os.listdir(batch_input_path)
                if name.lower().endswith((".json", ".jsonl"))
            ]
        )
        if len(candidates) == 0:
            raise ValueError(
                f"No .json/.jsonl files found in directory: {batch_input_path}"
            )
        if len(candidates) > 1:
            raise ValueError(
                "Multiple .json/.jsonl files found in directory. "
                "Please pass a single file path via --batch_process_dir: "
                f"{candidates}"
            )
        resolved_path = candidates[0]
        logger.info("Resolved --batch_process_dir directory to file: %s", resolved_path)

    if resolved_path.lower().endswith(".jsonl"):
        loaded = []
        with open(resolved_path, "r") as infile:
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                loaded.append(json.loads(line))
    else:
        loaded = json.load(open(resolved_path))

    if isinstance(loaded, dict) and "data" in loaded:
        entries = loaded["data"]
    elif isinstance(loaded, list):
        entries = loaded
    else:
        raise ValueError(
            "Unsupported input format for --batch_process_dir. "
            "Expected a list of response records or a dict with key 'data'."
        )

    normalized = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Entry {idx} is not a JSON object: {type(entry)}")
        # Accept rubric_eval-style rows by mapping answer_text -> output.
        if "output" not in entry and "answer_text" in entry:
            entry = dict(entry)
            entry["output"] = entry["answer_text"]
        if "input" not in entry:
            entry = dict(entry)
            entry["input"] = ""
        normalized.append(entry)

    return normalized


def record_lookup_keys(record):
    """Return stable identifiers useful for aligning sampled predictions to gold."""
    keys = []
    for field in ("id", "case_id", "idx"):
        value = str(record.get(field, "")).strip()
        if value:
            keys.append(f"{field}:{value}")
    for field in ("input", "initial_prompt", "question"):
        value = str(record.get(field, "")).strip()
        if value:
            keys.append(f"text:{value}")
            stripped = value.rstrip("?.!")
            if stripped != value:
                keys.append(f"text:{stripped}")
    return keys


def extract_gold_answer(record):
    """Extract a human reference answer when a dataset record has one."""
    for field in ("output", "answer", "reference_answer", "gold_answer"):
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def attach_gold_answers(responses, gold_answer_file, logger):
    """Attach gold answers by id/text when available; return True on full match."""
    gold_records = load_records_file(gold_answer_file)
    gold_by_key = {}
    ordered_gold_answers = []
    for record in gold_records:
        if not isinstance(record, dict):
            continue
        answer = extract_gold_answer(record)
        if not answer:
            continue
        ordered_gold_answers.append(answer)
        for key in record_lookup_keys(record):
            gold_by_key[key] = answer

    if not gold_by_key and not ordered_gold_answers:
        logger.warning(
            "Gold answer file has no usable output/answer fields. "
            "Switching to --no_reference mode."
        )
        return False

    attached = 0
    for index, response in enumerate(responses):
        answer = ""
        for key in record_lookup_keys(response):
            answer = gold_by_key.get(key, "")
            if answer:
                break
        if not answer and len(ordered_gold_answers) == len(responses):
            answer = ordered_gold_answers[index]
        if answer:
            response["answer"] = answer
            attached += 1

    if attached != len(responses):
        logger.warning(
            "Only attached %s/%s gold answers from %s. "
            "Switching to --no_reference mode.",
            attached,
            len(responses),
            gold_answer_file,
        )
        return False
    return True


def preprocess_text(text):
    """
    Clean up text: remove reference section, URLS, non-ascii chars
    """
    # clean up empty line
    paragraphs = text.split("\n")
    paragraphs = [i for i in paragraphs if len(i) > 0]
    # clean up section title and remove reference section
    cleaned_pargraphs = []
    for i in paragraphs:
        if i == "# References":
            break
        if i.startswith("#"):
            i = "section: " + i.replace("#", "").strip()
        cleaned_pargraphs.append(i)
    text = "\n".join(cleaned_pargraphs)
    # remove URLS
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    # remove non-ascii char
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    # remove citation bracket (e.g. [10])
    text = re.sub(r"\[\d+\]", "", text)
    # remove non alphanumeric char
    text = re.sub(r"[^\w\s]", "", text)
    return text


def get_conversation_prompt(filled_prompt):
    """
    From filled prompt, convert it into llama-2 conversation prompt
    """
    conv = get_conv_template("llama-2")
    conv.set_system_message("You are a fair evaluator language model.")
    conv.append_message(conv.roles[0], filled_prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return prompt


def _truncate_text_for_judge(text, max_chars):
    """Best-effort truncation helper used only as overflow fallback."""
    if max_chars is None:
        return text
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    suffix = "\n[TRUNCATED_FOR_JUDGE_CONTEXT]\n"
    keep = max(0, max_chars - len(suffix))
    return text[:keep] + suffix


def _is_context_overflow_error(exc):
    """Detect context-window overflow errors from OpenAI-compatible backends."""
    msg = str(exc).lower()
    overflow_markers = [
        "maximum context length",
        "requested 512 output tokens",
        "input_tokens",
        "this model's maximum context length",
        "total of at least",
    ]
    return any(marker in msg for marker in overflow_markers)


def format_prompt(
    instruction,
    response,
    rubric,
    no_reference=False,
    top_n=8,
    response_max_chars=None,
    reference_max_chars=None,
    doc_text_max_chars=None,
):
    """
    Fill prompt_template with rubric and response
    """
    output = _truncate_text_for_judge(response.get("output", ""), response_max_chars)
    if "final_passages" in response:
        final_passages = _truncate_text_for_judge(
            response.get("final_passages", ""), reference_max_chars
        )
        text = "References:\n" + final_passages
        output += text
    else:
        if "docs" not in response and "ctxs" in response:
            response["docs"] = response["ctxs"][:top_n]

        if (
            rubric["use_title"] is True
            and "docs" in response
            and response["docs"] is False
        ):
            titles = "References:\n"
            for idx, doc in enumerate(response["docs"]):
                if "[{0}]".format(idx + 1) in output:
                    titles += "[{0}] {1}\n".format(idx + 1, doc["title"])

        if (
            rubric["use_title"] is True
            and "docs" in response
            and response["docs"] is True
        ):
            text = "References:\n"
            for idx, doc in enumerate(response["docs"]):
                if "[{0}]".format(idx + 1) in output:
                    titles += "[{0}] {1}\n".format(idx + 1, doc["title"])
                    text += "[{0}] {1}: {2}\n".format(
                        idx + 1,
                        doc["title"],
                        _truncate_text_for_judge(
                            doc.get("text", ""), doc_text_max_chars
                        ),
                    )
            output += text

    if no_reference is False:
        entry = {
            "instruction": instruction + " " + response.get("input", ""),
            "response": output,
            "reference_answer": response["answer"],
        }
    else:
        entry = {
            "instruction": instruction + " " + response.get("input", ""),
            "response": output,
        }

    entry.update(rubric)
    if no_reference is False:
        if args.without_header is True:
            filled_prompt = prompt_template_wo_header.format(**entry)
        else:
            filled_prompt = prompt_template.format(**entry)
    else:
        filled_prompt = prompt_template_no_reference.format(**entry)

    return get_conversation_prompt(filled_prompt)


def get_grading_dict(
    responses,
    instruction,
    model,
    rubric_path="rubrics/prometheus_rubrics_v8.json",
    disable_sample=False,
    temperature=0.01,
    top_p=0.95,
    max_new_tokens=512,
    repetition_penalty=1.03,
    logger=None,
    sampling_params=None,
    no_reference=False,
    top_n=8,
    aspects=None,
    litellm_cfg=None,
):
    grading = {}
    rubrics = read_json(rubric_path)

    # Read all files in the given directory
    for rubric_idx, rubric in enumerate(rubrics):
        if aspects is not None and rubric["aspect"] not in aspects:
            continue
        grading[rubric["criteria_description"]] = {}

        prompts = []
        for response_idx, response in enumerate(responses):
            # generate evaluation prompt and tokenize
            if logger is not None:
                logger.info(
                    f"processing for rubric {rubric_idx + 1}/{len(rubrics)}, response {response_idx + 1}/{len(responses)}, response length: {len(response.get('output', ''))}"
                )

            prompt = format_prompt(
                instruction=instruction,
                response=response,
                rubric=rubric,
                no_reference=no_reference,
                top_n=top_n,
            )
            prompts.append(prompt)

        if litellm_cfg is not None:
            try:
                from litellm import completion
            except ImportError as exc:
                raise ImportError(
                    "litellm is required for --model litellm_openai mode."
                ) from exc

            decoded_outputs = []
            for response_idx, prompt in enumerate(prompts):
                try:
                    response = completion(
                        model=litellm_cfg["model"],
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_new_tokens,
                        api_base=litellm_cfg["api_base"],
                        api_key=litellm_cfg["api_key"],
                    )
                    decoded_outputs.append(response.choices[0].message.content)
                    continue
                except Exception as exc:
                    if not _is_context_overflow_error(exc):
                        raise
                    if logger is not None:
                        logger.warning(
                            "Judge overflow on response %s; retrying with truncated prompt.",
                            response_idx,
                        )

                recovered = False
                # Progressive fallback: truncate only the problematic entry.
                # Keep retries local so evaluation continues for the rest.
                fallback_plans = [
                    {
                        "response_max_chars": 12000,
                        "top_n": min(top_n, 6),
                        "max_new_tokens": min(max_new_tokens, 384),
                    },
                    {
                        "response_max_chars": 9000,
                        "top_n": min(top_n, 4),
                        "max_new_tokens": min(max_new_tokens, 320),
                    },
                    {
                        "response_max_chars": 7000,
                        "top_n": min(top_n, 3),
                        "max_new_tokens": min(max_new_tokens, 256),
                    },
                    {
                        "response_max_chars": 5000,
                        "top_n": min(top_n, 2),
                        "max_new_tokens": min(max_new_tokens, 192),
                    },
                    {
                        "response_max_chars": 3500,
                        "top_n": 1,
                        "max_new_tokens": min(max_new_tokens, 128),
                    },
                ]

                for plan in fallback_plans:
                    retry_prompt = format_prompt(
                        instruction=instruction,
                        response=responses[response_idx],
                        rubric=rubric,
                        no_reference=no_reference,
                        top_n=plan["top_n"],
                        response_max_chars=plan["response_max_chars"],
                        reference_max_chars=4000,
                        doc_text_max_chars=1200,
                    )
                    try:
                        retry_response = completion(
                            model=litellm_cfg["model"],
                            messages=[{"role": "user", "content": retry_prompt}],
                            temperature=temperature,
                            top_p=top_p,
                            max_tokens=plan["max_new_tokens"],
                            api_base=litellm_cfg["api_base"],
                            api_key=litellm_cfg["api_key"],
                        )
                        decoded_outputs.append(
                            retry_response.choices[0].message.content
                        )
                        recovered = True
                        if logger is not None:
                            logger.warning(
                                "Recovered judge overflow on response %s with response_max_chars=%s, top_n=%s, max_new_tokens=%s.",
                                response_idx,
                                plan["response_max_chars"],
                                plan["top_n"],
                                plan["max_new_tokens"],
                            )
                        break
                    except Exception as retry_exc:
                        if not _is_context_overflow_error(retry_exc):
                            raise
                        continue

                if not recovered:
                    if logger is not None:
                        logger.error(
                            "Judge overflow persists on response %s after truncation retries; assigning fallback score.",
                            response_idx,
                        )
                    decoded_outputs.append(
                        "Feedback: Context overflow prevented reliable grading for this entry. [RESULT] 3"
                    )

        elif sampling_params is not None:
            # vLLM generation
            outputs = model.generate(prompts, sampling_params)
            decoded_outputs = [output.outputs[0].text for output in outputs]
        else:
            # Standard Transformers generation
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                model.config._name_or_path
                if hasattr(model, "config")
                else "kaist-ai/prometheus-7b-v1.0"
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            decoded_outputs = []
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        do_sample=(not disable_sample),
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                # Keep only newly generated tokens
                new_tokens = output_ids[0][inputs.input_ids.shape[1] :]
                decoded_outputs.append(
                    tokenizer.decode(new_tokens, skip_special_tokens=True)
                )

        for response_idx, (response, decoded_output) in enumerate(
            zip(responses, decoded_outputs)
        ):
            score = decoded_output[
                decoded_output.find("[RESULT] ") + len("[RESULT] ") :
            ].split("\n")[0]
            print(score)
            feedback = (
                decoded_output[decoded_output.find("Feedback: ") + len("Feedback: ") :]
                if "Feedback: " in decoded_output
                else decoded_output
            )
            try:
                int(score)
            except Exception:
                pattern = r"the overall score is (\d+)"
                match = re.search(pattern, feedback)
                if match:
                    score = match.group(1)
            print("final score")
            print(score)
            grading[rubric["criteria_description"]][response_idx] = {
                "feedback": feedback,
                "score": score,
            }
    return grading


def extract_score(result):
    result_dics = {}
    for k in result:
        scores = []
        for i in list(result[k].values()):
            if i["score"] in ["1", "2", "3", "4", "5"]:
                scores.append(int(i["score"]))
            elif i["score"].split("\n\n")[0] in ["1", "2", "3", "4", "5"]:
                scores.append(int(i["score"].split("\n\n")[0]))
        result_dics[k] = np.mean(scores)
        # result_dics[k + "_std"] = np.std(scores)
    result_dics["average"] = np.mean(list(result_dics.values()))
    return result_dics


def main(args):
    litellm_cfg = None

    # Loading evaluator LM
    if args.model == "litellm_openai":
        if not args.litellm_api_base:
            raise ValueError(
                "--litellm_api_base is required when --model litellm_openai. "
                "You can also set LLM_BASE_URL or JUDGE_LLM_BASE_URL."
            )
        if not args.litellm_model:
            raise ValueError(
                "--litellm_model is required when --model litellm_openai. "
                "You can also set JUDGE_LLM_MODEL or LLM_MODEL."
            )

        api_base = args.litellm_api_base.strip()
        # Common typo guard: convert `http:127.0.0.1:8000/v1` -> `http://127.0.0.1:8000/v1`.
        if api_base.startswith("http:") and not api_base.startswith("http://"):
            api_base = "http://" + api_base[len("http:") :].lstrip("/")
        if api_base.startswith("https:") and not api_base.startswith("https://"):
            api_base = "https://" + api_base[len("https:") :].lstrip("/")

        parsed_api_base = urlparse(api_base)
        if (
            parsed_api_base.scheme not in {"http", "https"}
            or not parsed_api_base.netloc
        ):
            raise ValueError(
                "Invalid --litellm_api_base. Expected a full URL such as "
                "http://127.0.0.1:8000/v1"
            )

        resolved_model = (
            args.litellm_model
            if "/" in args.litellm_model
            else f"openai/{args.litellm_model}"
        )
        litellm_cfg = {
            "api_base": api_base,
            "api_key": args.litellm_api_key,
            "model": resolved_model,
        }
        logger.info(
            "Using OpenAI-compatible remote judge via LiteLLM: base=%s model=%s",
            litellm_cfg["api_base"],
            litellm_cfg["model"],
        )
        model = None
        sampling_params = None

    elif args.load_vllm is True:
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "vllm is not installed. Please install it to use --load_vllm. Note: vllm is not natively supported on Mac."
            )
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
        )
        # Resolve HF model ID to an already-cached local snapshot path so that
        # vLLM does not attempt to re-download the weights and exhaust disk quota.
        model_path = args.model
        try:
            from huggingface_hub import snapshot_download

            local_path = snapshot_download(
                repo_id=args.model,
                cache_dir=args.download_dir,
                local_files_only=True,
            )
            model_path = local_path
            logger.info("Using locally cached model at: %s", model_path)
        except Exception as e:
            logger.warning(
                "Could not resolve local cache path (%s); vLLM will download the model. "
                "Set --download_dir to a directory with sufficient disk space.",
                e,
            )
        model = LLM(
            model_path,
            download_dir=args.download_dir,
            tokenizer_mode="auto",
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=True,
            disable_custom_all_reduce=True,
            quantization="fp8",
        )
    else:
        model = LlamaForCausalLM.from_pretrained(args.model, device_map="auto")

    # Load responses
    responses = load_batch_results(args.batch_process_dir)

    # Load human-written references when the selected dataset has them. Some
    # SQA_bench question-only files are JSONL and intentionally do not include
    # reference answers; those runs can still produce no-reference judge scores.
    if args.no_reference is False and args.gold_answer_file is not None:
        attached_gold = attach_gold_answers(
            responses, args.gold_answer_file, logger=logger
        )
        if not attached_gold:
            args.no_reference = True

    if args.no_reference is False:
        missing_answer = any("answer" not in response for response in responses)
        if missing_answer:
            logger.warning(
                "Responses do not contain complete reference answers. "
                "Switching to --no_reference mode automatically."
            )
            args.no_reference = True

    if args.self_consistency is True:
        grading_dict = {}
        for iter_i in range(3):
            print("start grading: {0}".format(iter_i))
            grading = get_grading_dict(
                responses=responses,
                instruction=args.instruction,
                model=model,
                rubric_path=args.rubric_path,
                disable_sample=args.disable_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                repetition_penalty=args.repetition_penalty,
                logger=logger,
                sampling_params=sampling_params if args.load_vllm else None,
                no_reference=args.no_reference,
                top_n=args.top_n,
                litellm_cfg=litellm_cfg,
            )
            grading_dict[iter_i] = grading
        final_grading = {}
        for aspect in grading_dict[iter_i].keys():
            final_grading[aspect] = {}
            for instance in grading_dict[iter_i][aspect]:
                if args.most_common is True:
                    c = Counter(
                        [grading_dict[i][aspect][instance]["score"] for i in range(3)]
                    )
                    final_decision = c.most_common()[0][0]
                    final_grading[aspect][instance] = {"score": final_decision}
                else:
                    valid_choices = []
                    for i in range(3):
                        pred = grading_dict[i][aspect][instance]["score"]
                        try:
                            pred = int(pred)
                            valid_choices.append(pred)
                        except Exception:
                            print("conversion error")

                    final_decision = np.mean(valid_choices)
                    final_grading[aspect][instance] = {"score": final_decision}
        grading = final_grading

    else:
        grading = get_grading_dict(
            responses=responses,
            instruction=args.instruction,
            model=model,
            rubric_path=args.rubric_path,
            disable_sample=args.disable_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
            logger=logger,
            sampling_params=sampling_params if args.load_vllm else None,
            no_reference=args.no_reference,
            top_n=args.top_n,
            aspects=args.aspects,
            litellm_cfg=litellm_cfg,
        )

    summary_result = extract_score(grading)
    grading["summary"] = summary_result

    print(summary_result)

    # Merge with existing file if it exists so we don't overwrite during multi-step runs
    output_file = os.path.join(args.output_path, "results.json")
    if os.path.exists(output_file):
        try:
            with open(output_file, "r") as infile:
                existing_data = json.load(infile)
                # merge case specific results
                for case_id, aspects in grading.items():
                    if case_id == "summary":
                        continue
                    if case_id not in existing_data:
                        existing_data[case_id] = {}
                    existing_data[case_id].update(aspects)

                # merge summary, then recompute it from the merged aspects so
                # multi-pass runs do not keep a stale "average" from an
                # earlier partial pass.
                merged_aspects = {
                    key: value
                    for key, value in existing_data.items()
                    if key != "summary" and isinstance(value, dict)
                }
                existing_data["summary"] = extract_score(merged_aspects)
                grading = existing_data
        except Exception as e:
            logger.warning(
                "Could not merge with existing results.json: %s. Overwriting.", e
            )

    # Save grading dictionary to output path
    with open(output_file, "w") as outfile:
        json.dump(grading, outfile, indent=2)
        logger.info("Grading complete. Output saved to: %s", args.output_path)


if __name__ == "__main__":
    global logger
    parser = argparse.ArgumentParser(description="Process some files.")
    parser.add_argument(
        "-b",
        "--batch_process_dir",
        required=True,
        help=(
            "Path to input responses (.json or .jsonl), or a directory containing one such file"
        ),
    )
    parser.add_argument("-f", "--gold_answer_file", help="Gold answer")
    parser.add_argument(
        "-o", "--output_path", required=True, help="Path to save the output JSON file"
    )
    parser.add_argument(
        "-i",
        "--instruction",
        default="Given a paper abstract, generate the Related Work section summarizing relevant papers.",
        help="Topic of the script your going to analyze",
    )

    parser.add_argument(
        "--rubric_path", default="eval_rubric_5.json", help="path to rubric json file"
    )

    parser.add_argument("--tokenizer", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument(
        "--model",
        default="kaist-ai/prometheus-13b-v1.0",
        help=(
            "Model to use (HF id), or 'litellm_openai' to call an OpenAI-compatible "
            "endpoint via LiteLLM."
        ),
    )
    parser.add_argument(
        "--litellm_api_base",
        type=str,
        default=os.environ.get(
            "JUDGE_LLM_BASE_URL", os.environ.get("LLM_BASE_URL", "")
        ),
        help="OpenAI-compatible base URL used when --model litellm_openai.",
    )
    parser.add_argument(
        "--litellm_api_key",
        type=str,
        default=os.environ.get(
            "JUDGE_LLM_API_KEY", os.environ.get("OPENAI_API_KEY", "EMPTY")
        ),
        help="API key used when --model litellm_openai.",
    )
    parser.add_argument(
        "--litellm_model",
        type=str,
        default=os.environ.get("JUDGE_LLM_MODEL", os.environ.get("LLM_MODEL", "")),
        help="Remote model name for LiteLLM OpenAI mode (e.g. openai/gpt-4o-mini or raw model id).",
    )
    parser.add_argument(
        "--disable_sample",
        action="store_true",
        help="Whether to disable sampling; default is False",
    )
    parser.add_argument(
        "--load_vllm",
        action="store_true",
        help="Load checkpoints via vllm for inference efficiency.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.01,
        help="Temperature for generation; default is 0.01",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top P for generation; default is 0.95",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate; default is 512",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.85")),
        help=(
            "vLLM GPU memory utilization target in [0,1]. "
            "Lower this if startup fails due to insufficient free GPU memory."
        ),
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.03,
        help="Repetition penalty; default is 1.03",
    )
    parser.add_argument(
        "--no_reference", action="store_true", help="whether to use reference or not."
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=8,
    )
    parser.add_argument("--aspects", type=str, nargs="+")
    parser.add_argument("--self_consistency", action="store_true")
    parser.add_argument("--most_common", action="store_true")
    _hf_home = os.environ.get(
        "HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    )
    parser.add_argument(
        "--download_dir", type=str, default=os.path.join(_hf_home, "hub")
    )
    parser.add_argument("--without_header", action="store_true")
    args = parser.parse_args()

    logger = logging.getLogger(__name__)

    assert os.path.exists(args.batch_process_dir), (
        f"batch_process_dir: {args.batch_process_dir} not exists"
    )
    output_directory = args.output_path
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)
        logger.info("Created directory: %s", output_directory)

    main(args)
