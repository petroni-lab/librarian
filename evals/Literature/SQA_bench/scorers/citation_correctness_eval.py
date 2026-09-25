import argparse
import collections
import gc
import json
import re
import string
import torch
import copy
from pathlib import Path

from nltk import sent_tokenize
import numpy as np
from rouge_score import rouge_scorer, scoring
from tqdm import tqdm
import logging
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from run_utils import load_jsonlines

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

GOOGLE_AUTOAIS_MODEL = "google/t5_xxl_true_nli_mixture"
OSU_AUTOAIS_MODEL = "osunlp/attrscore-flan-t5-xl"
input_prompt = "As an Attribution Validator, your task is to verify whether a given reference can support the given claim. A claim can be either a plain sentence or a question followed by its answer. Specifically, your response should clearly indicate the relationship: Attributable, Contradictory or Extrapolatory. A contradictory error occurs when you can infer that the answer contradicts the fact presented in the context, while an extrapolatory error means that you cannot infer the correctness of the answer based on the information provided in the context. \n\nClaim: {claim}\n Reference: {output}"

global autoais_model, autoais_tokenizer
global claim_autoais_model, claim_autoais_tokenizer
autoais_model, autoais_tokenizer = None, None
claim_autoais_model, claim_autoais_tokenizer = None, None
QUIET = False
AUTOAIS_MAX_INPUT_TOKENS = None
AUTOAIS_OOM_FALLBACK_TOKEN_LIMITS = [2048, 1024, 768, 512]
AUTOAIS_ENABLE_OOM_FALLBACK = True
ABSTRACT_HEADER = "Abstract:\n"
FULL_TEXT_HEADER = "Full Text Excerpt (Methods/Results when available):\n"
FULL_TEXT_SPLIT_MARKER = "\n\n" + FULL_TEXT_HEADER


def qprint(*args, **kwargs):
    if not QUIET:
        print(*args, **kwargs)


def _safe_div(numerator, denominator):
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _harmonic_mean(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _autoais_source_row_index(item, fallback):
    try:
        return int(item.get("_autoais_source_row_index", fallback))
    except (TypeError, ValueError):
        return fallback


def _is_cuda_oom_error(err):
    if isinstance(err, torch.OutOfMemoryError):
        return True
    return "out of memory" in str(err).lower()


def _cleanup_cuda_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# [1], [1, 2], [REF_1], and the snippet-level [1.1] / [1.4, 1.5] the synthesis
# agent has emitted since commit 05c3d9a. Without the optional `.k` group 57% of
# cited sentences in a 2026-09-04 scholarqa_bio run parsed as having NO citation
# and were scored unsupported. [n.k] is a refinement of [n] -- passage k of paper
# n -- so it collapses to paper n here, which is what the API contract tells a
# paper-level client to do.
CITATION_PATTERN = r"\[((?:REF_)?\d+(?:\.\d+)?(?:\s*,\s*(?:REF_)?\d+(?:\.\d+)?)*)\]"


def extract_citations(text):
    matches = re.findall(CITATION_PATTERN, text)

    citations = []
    for match in matches:
        for raw_ref in match.split(","):
            raw_ref = raw_ref.strip()
            raw_ref = re.sub(r"^REF_", "", raw_ref)
            # Drop the passage index: [3.2] cites paper 3.
            paper_ref = "[{}]".format(int(raw_ref.split(".")[0]))
            # One paper cited through several of its passages ([1.1, 1.4]) is one
            # citation, not two. Duplicates would double-count the precision
            # denominator and duplicate the passage text in the joint entailment.
            if paper_ref not in citations:
                citations.append(paper_ref)

    return citations


def remove_citations(text):
    # Same grammar as CITATION_PATTERN, uncaptured: a [1.1] left in the text
    # would otherwise reach the NLI model as literal noise.
    citation_pattern = r"\[(?:REF_)?\d+(?:\.\d+)?(?:\s*,\s*(?:REF_)?\d+(?:\.\d+)?)*\]"
    # Remove all citations from the text
    cleaned_text = re.sub(citation_pattern, "", text)
    # Optionally, remove extra spaces that might result from removing citations
    cleaned_text = re.sub(r"\s{2,}", " ", cleaned_text).strip()
    cleaned_text = cleaned_text.replace(" .", ".")
    cleaned_text = cleaned_text.replace(" ,", ",")
    return cleaned_text


def get_max_memory():
    """Get the maximum memory available for the current GPU for loading models."""
    if not torch.cuda.is_available():
        return None  # device_map="auto" will handle CPU/MPS
    free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024**3)
    max_memory = f"{free_in_GB - 6}GB"
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    return max_memory


def extract_citation_indices(text):
    """Extract citation indices as zero-based integer ids from bracket citations."""

    refs = []
    for citation in extract_citations(text):
        refs.append(int(citation[1:-1]) - 1)
    return refs


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(a_gold, a_pred):
    """Compute F1 score between two strings."""

    def _get_tokens(s):
        if not s:
            return []
        return normalize_answer(s).split()

    gold_toks = _get_tokens(a_gold)
    pred_toks = _get_tokens(a_pred)

    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def compute_exact(a_gold, a_pred):
    """Check whether two strings are equal up to normalization."""

    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def exact_presence(short_answers, context):
    """Verify if any of the answers is present in the given context.
    Args:
        short_answers: list of short answers to look for in the context
        context: a paragraph to search for short answers
    Returns:
        true if any of the short answers is present in the context
    """

    n_short_answers = [normalize_answer(sa) for sa in short_answers]
    n_context = normalize_answer(context)

    for ans in n_short_answers:
        if ans in n_context:
            return True

    return False


def compute_rouge(data):
    """Main function for rouge scoring.
    If two references are provided,
    the best score is chosen for each instance.
    Args:
        data: requires field `output` and `answer` (or `annotations` for ASQA)
        metrics: list of evaluation metrics
    Returns:
        dictionary representation of rouge scores
    """

    def _rouge_calculation(
        hypotheses, references1, references2=[], metrics=["rougeL", "rouge1", "rouge2"]
    ):
        if references2 == []:
            references2 = references1

        scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
        aggregator = scoring.BootstrapAggregator()
        all_scores = []

        for i in range(len(hypotheses)):
            scores1 = scorer.score(references1[i], hypotheses[i])
            scores2 = scorer.score(references2[i], hypotheses[i])
            if scores1["rougeL"].fmeasure > scores2["rougeL"].fmeasure:
                aggregator.add_scores(scores1)
                all_scores.append(scores1["rougeL"].fmeasure)
            else:
                aggregator.add_scores(scores2)
                all_scores.append(scores2["rougeL"].fmeasure)

        scores = {m: [] for m in metrics}

        for m in metrics:
            fmeasure = aggregator.aggregate()[m].mid.fmeasure
            scores[m].append(fmeasure)

        for m in scores:
            scores[m] = 100 * sum(scores[m]) / len(scores[m])

        return scores, all_scores

    hypotheses = {}
    references1 = {}
    references2 = {}

    for idx, item in enumerate(data):
        hypotheses[idx] = item["output"]
        if "annotations" in item and item["annotations"] is not None:  # For ASQA
            references1[idx] = item["annotations"][0]["long_answer"]
            references2[idx] = item["annotations"][1]["long_answer"]
        else:
            references1[idx] = item["answer"]
            references2[idx] = item["answer"]

    h, r1, r2 = [], [], []

    for key in references1:
        h.append(hypotheses[key])
        r1.append(references1[key])

        if references2 is not None:
            r2.append(references2[key])

    h = ["\n".join(sent_tokenize(text.lower())) for text in h]
    r1 = ["\n".join(sent_tokenize(text.lower())) for text in r1]
    r2 = ["\n".join(sent_tokenize(text.lower())) for text in r2]
    scores, all_scores = _rouge_calculation(h, r1, r2)

    qprint(scores["rougeL"])
    return scores, all_scores


def compute_str_em(data):
    """Compute STR-EM metric (only for ASQA)
    Args:
        data: requires field `qa_pairs/short_answers` and `output`
    Returns:
        STR-EM and STR-EM-HIT ()
    """

    if "qa_pairs" not in data[0] or data[0]["qa_pairs"] is None:
        return 0, 0

    acc = []
    hit = []

    for item in data:
        loc_acc = []
        for qa_pair in item["qa_pairs"]:
            loc_acc.append(exact_presence(qa_pair["short_answers"], item["output"]))
        acc.append(np.mean(loc_acc))
        hit.append(int(np.mean(loc_acc) == 1))

    return 100 * np.mean(acc), 100 * np.mean(hit)


def compute_len(data):
    """Compute average length of predictions."""

    res, cntr = 0, 0
    for item in data:
        res += len(item["output"].split())
        cntr += 1
    return res / cntr


def compute_single_qa(data):
    """Compute QA-based accuracy.
    Args:
        data: requires filed `qa_pairs/short_answers` and `output`
    Returns:
        QA metrics (QA-EM, QA-F1, QA-Hit)
    """

    # Get prediction
    em, f1 = [], []
    for item in tqdm(data):
        answers = [item["answer"]]
        prediction = item["output"]
        qprint(answers)
        qprint(prediction)
        em.append([compute_exact(a, prediction) for a in answers])
        f1.append([compute_f1(a, prediction) for a in answers])
        qprint(em[-1])

    return {
        "QA-EM": 100 * np.mean(em),
        "QA-F1": 100 * np.mean(f1),
    }


def compute_match(data):
    """Compute QA-based accuracy.
    Args:
        data: requires filed `qa_pairs/short_answers` and `output`
    Returns:
        QA metrics (QA-EM, QA-F1, QA-Hit)
    """

    # Get prediction
    match = []
    for item in tqdm(data):
        answers = remove_citations(item["answer"]).lower()
        prediction = remove_citations(item["output"]).lower()
        if answers in prediction:
            match.append(1.0)
        else:
            match.append(0.0)

    return {
        "match": 100 * np.mean(match),
    }


def _run_nli_autoais(passage, claim):
    """
    Run inference for assessing AIS between a premise and hypothesis.
    Adapted from https://github.com/google-research-datasets/Attributed-QA/blob/main/evaluation.py
    """
    # global autoais_model, autoais_tokenizer
    # input_text = "premise: {} hypothesis: {}".format(passage, claim)
    # input_ids = autoais_tokenizer(input_text, return_tensors="pt").input_ids.to(autoais_model.device)
    # with torch.inference_mode():
    #     outputs = autoais_model.generate(input_ids, max_new_tokens=10)
    # result = autoais_tokenizer.decode(outputs[0], skip_special_tokens=True)
    # inference = 1 if result == "1" else 0
    global claim_autoais_model, claim_autoais_tokenizer
    global AUTOAIS_MAX_INPUT_TOKENS
    global AUTOAIS_OOM_FALLBACK_TOKEN_LIMITS
    global AUTOAIS_ENABLE_OOM_FALLBACK

    input_text = input_prompt.format_map({"output": passage, "claim": claim})

    def _generate_with_limit(token_limit=None):
        tokenizer_kwargs = {"return_tensors": "pt"}
        if token_limit is not None:
            tokenizer_kwargs["truncation"] = True
            tokenizer_kwargs["max_length"] = token_limit

        input_ids = claim_autoais_tokenizer(
            input_text, **tokenizer_kwargs
        ).input_ids.to(claim_autoais_model.device)
        with torch.inference_mode():
            outputs = claim_autoais_model.generate(
                input_ids, max_new_tokens=10, use_cache=False
            )
        return outputs

    token_limit = AUTOAIS_MAX_INPUT_TOKENS
    try:
        outputs = _generate_with_limit(token_limit=token_limit)
    except (RuntimeError, torch.OutOfMemoryError) as err:
        is_cuda_oom = _is_cuda_oom_error(err)
        if not (AUTOAIS_ENABLE_OOM_FALLBACK and is_cuda_oom):
            raise

        last_error_message = str(err)
        del err
        logger.warning(
            "CUDA OOM in AutoAIS NLI call. Retrying with progressively smaller token limits."
        )
        _cleanup_cuda_memory()

        fallback_limits = [
            limit
            for limit in AUTOAIS_OOM_FALLBACK_TOKEN_LIMITS
            if token_limit is None or limit < token_limit
        ]
        if token_limit is not None and token_limit not in fallback_limits:
            fallback_limits = [token_limit] + fallback_limits

        outputs = None
        for fallback_limit in fallback_limits:
            try:
                outputs = _generate_with_limit(token_limit=fallback_limit)
                logger.warning(
                    "AutoAIS recovered from OOM using truncation limit %d tokens.",
                    fallback_limit,
                )
                break
            except (RuntimeError, torch.OutOfMemoryError) as fallback_err:
                if not _is_cuda_oom_error(fallback_err):
                    raise
                last_error_message = str(fallback_err)
                del fallback_err
                _cleanup_cuda_memory()

        if outputs is None:
            raise RuntimeError(last_error_message)

    result = claim_autoais_tokenizer.decode(outputs[0], skip_special_tokens=True)
    if result == "Attributable":
        inference = 1.0
    else:
        inference = 0.0

    return inference


def _format_document(doc, text_override=None):
    """Format document for AutoAIS."""
    if "sent" in doc:
        return "Title: %s\n%s" % (doc["title"], doc["sent"])

    doc_text = text_override if text_override is not None else doc.get("text", "")
    if "title" in doc:
        return "Title: %s\n%s" % (doc["title"], doc_text)
    return doc_text


def _document_source_variants(doc):
    """Split one paper-level ctx text into abstract/full-text variants when possible."""
    if "sent" in doc:
        return {"combined": _format_document(doc)}

    raw_text = str(doc.get("text", "") or "")
    variants = {"combined": _format_document(doc, raw_text)}
    # Only split when the combined serializer actually inserted a second block.
    if FULL_TEXT_SPLIT_MARKER not in raw_text:
        return variants

    abstract_part, full_text_part = raw_text.split(FULL_TEXT_SPLIT_MARKER, 1)
    if abstract_part.startswith(ABSTRACT_HEADER):
        abstract_part = abstract_part[len(ABSTRACT_HEADER) :]

    abstract_part = abstract_part.strip()
    full_text_part = full_text_part.strip()
    if abstract_part:
        variants["abstract"] = _format_document(doc, abstract_part)
    if full_text_part:
        variants["full_text_excerpt"] = _format_document(doc, full_text_part)
    return variants


def _build_joint_passage(docs, ref_ids, source_mode):
    """Build a joint passage from cited docs using one evidence source mode."""
    passages = []
    for psgs_id in ref_ids:
        if psgs_id < 0 or psgs_id >= len(docs):
            continue
        variants = _document_source_variants(docs[psgs_id])
        if source_mode == "abstract":
            passage = variants.get("abstract") or variants.get("combined", "")
        elif source_mode == "full_text_excerpt":
            passage = variants.get("full_text_excerpt") or variants.get("combined", "")
        else:
            passage = variants.get("combined", "")
        if passage:
            passages.append(passage)
    return "\n".join(passages)


def _run_autoais_with_document_variants(docs, ref_ids, claim):
    """Try abstract-only, then excerpt-only, then combined evidence."""
    last_passage = ""
    for source_mode in ("abstract", "full_text_excerpt", "combined"):
        joint_passage = _build_joint_passage(docs, ref_ids, source_mode)
        if not joint_passage:
            continue
        last_passage = joint_passage
        if _run_nli_autoais(joint_passage, claim):
            return 1.0, source_mode, joint_passage
    return 0.0, "combined", last_passage


def compute_autoais(data, at_most_citations=None, progress_label=None):
    """
    Compute AutoAIS score.

    Args:
        data: requires field `output` and `docs`
              - docs should be a list of items with fields `title` and `text` (or `phrase` and `sent` for QA-extracted docs)
        citation: check citations and use the corresponding references.
    """

    global claim_autoais_model, claim_autoais_tokenizer

    if claim_autoais_model is None:
        logger.info("Loading Claims AutoAIS model...")
        claim_autoais_model = AutoModelForSeq2SeqLM.from_pretrained(
            OSU_AUTOAIS_MODEL,
            torch_dtype=torch.bfloat16,
            max_memory=get_max_memory(),
            device_map="auto",
        )
        # extra_special_tokens={} overrides the list stored in this model's old
        # tokenizer_config.json: transformers >= 4.53 expects a dict there and
        # otherwise crashes in _set_model_specific_special_tokens ('list' object
        # has no attribute 'keys'). The <extra_id_*> sentinels live in the
        # SentencePiece model, so dropping this kwarg is harmless for scoring.
        claim_autoais_tokenizer = AutoTokenizer.from_pretrained(
            OSU_AUTOAIS_MODEL, use_fast=False, extra_special_tokens={}
        )

    logger.info("Running AutoAIS...")

    ais_scores = []
    ais_scores_prec = []
    ais_scores_f1 = []
    ais_score_row_indices = []

    rec_supported_total = 0.0
    rec_total_total = 0
    prec_supported_total = 0.0
    prec_total_total = 0

    sent_total = 0
    sent_mcite = 0
    sent_mcite_support = 0
    sent_mcite_overcite = 0
    autoais_log = []
    cited_paper_total = []
    for row_index, item in enumerate(tqdm(data, desc=progress_label)):
        sents = sent_tokenize(item["output"])
        if len(sents) == 0:
            continue

        target_sents = [remove_citations(sent).strip() for sent in sents]

        cited_papers = set(extract_citations(item["output"]))
        cited_paper_total.append(len(cited_papers))

        entail = 0
        entail_prec = 0
        total_citations = 0
        total_sents = 0
        previous_citations = None
        citations = item["ctxs"]
        for sent_id, sent in enumerate(sents):
            # add minimum length for citation
            if len(sent) < 50:
                continue
            total_sents += 1

            target_sent = target_sents[
                sent_id
            ]  # Citation removed and (if opted for) decontextualized
            joint_entail = -1  # Undecided

            # Find references
            ref = extract_citation_indices(sent)  # In text citation id starts from 1
            # ref = [int(r[1:]) for r in re.findall(r"\[\d+", sent)]
            logger.info(f"For `{sent}`, find citations {ref}")
            if len(ref) == 0 and previous_citations is not None:
                ref = previous_citations

            if len(ref) == 0:
                # No citations
                joint_entail = 0
            elif any([ref_id >= len(citations) for ref_id in ref]):
                # Citations out of range
                joint_entail = 0
            else:
                previous_citations = ref
                if at_most_citations is not None:
                    ref = ref[:at_most_citations]
                total_citations += len(ref)
                joint_passage = _build_joint_passage(item["docs"], ref, "combined")

            # If not directly rejected by citation format error, calculate the recall score
            if joint_entail == -1:
                joint_entail, support_source, joint_passage = (
                    _run_autoais_with_document_variants(item["docs"], ref, target_sent)
                )
                autoais_log.append(
                    {
                        "question": item["question"],
                        "output": item["output"],
                        "claim": sent,
                        "passage": [joint_passage],
                        "support_source": support_source,
                        "model_type": "NLI",
                        "model_output": joint_entail,
                    }
                )

            entail += joint_entail
            if len(ref) > 1:
                sent_mcite += 1

            # calculate the precision score if applicable
            if joint_entail and len(ref) > 1:
                sent_mcite_support += 1
                # Precision check: did the model cite any unnecessary documents?
                for psgs_id in ref:
                    # condition A
                    nli_result, _, _ = _run_autoais_with_document_variants(
                        item["docs"], [psgs_id], target_sent
                    )

                    # condition B
                    if not nli_result:
                        subset_exclude = copy.deepcopy(ref)
                        subset_exclude.remove(psgs_id)
                        nli_result, _, _ = _run_autoais_with_document_variants(
                            item["docs"], subset_exclude, target_sent
                        )
                        if nli_result:  # psgs_id is not necessary
                            sent_mcite_overcite += 1
                        else:
                            entail_prec += 1
                    else:
                        entail_prec += 1
            else:
                entail_prec += joint_entail

        # Track totals across the whole run for the multi-citation summary.
        sent_total += total_sents
        rec_supported_total += entail
        rec_total_total += total_sents
        prec_supported_total += entail_prec
        prec_total_total += total_citations

        if total_sents > 0:
            rec_i = entail / total_sents
        else:
            rec_i = 0
        ais_scores.append(rec_i)

        prec_i = entail_prec / total_citations if total_citations > 0 else 0
        ais_scores_prec.append(prec_i)
        ais_scores_f1.append(_harmonic_mean(prec_i, rec_i))
        ais_score_row_indices.append(_autoais_source_row_index(item, row_index))

    if sent_total > 0 and sent_mcite > 0:
        sent_mcite_support_pct = (
            100 * sent_mcite_support / sent_mcite if sent_mcite > 0 else 0
        )
        sent_mcite_overcite_pct = (
            100 * sent_mcite_overcite / sent_mcite_support
            if sent_mcite_support > 0
            else 0
        )
        qprint(
            "Among all sentences, %.2f%% have multiple citations, among which %.2f%% are supported by the joint set, among which %.2f%% overcite."
            % (
                100 * sent_mcite / sent_total,
                sent_mcite_support_pct,
                sent_mcite_overcite_pct,
            )
        )

    citation_rec = _safe_div(rec_supported_total, rec_total_total)
    citation_prec = _safe_div(prec_supported_total, prec_total_total)
    citation_f1 = _harmonic_mean(citation_prec, citation_rec)

    return {
        "citation_rec": 100 * citation_rec,
        "citation_rec_all": ais_scores,
        "citation_prec": 100 * citation_prec,
        "citation_prec_all": ais_scores_prec,
        "citation_f1": 100 * citation_f1,
        "citation_f1_all": ais_scores_f1,
        "citation_row_indices_all": ais_score_row_indices,
        "cited_paper_numbers": np.mean(cited_paper_total),
        "citation_rec_supported_total": rec_supported_total,
        "citation_rec_total_total": rec_total_total,
        "citation_prec_supported_total": prec_supported_total,
        "citation_prec_total_total": prec_total_total,
    }


def compute_autoais_short_form(data, at_most_citations=None, progress_label=None):
    """
    Compute AutoAIS score.

    Args:
        data: requires field `output` and `docs`
              - docs should be a list of items with fields `title` and `text` (or `phrase` and `sent` for QA-extracted docs)
        citation: check citations and use the corresponding references.
        decontext: decontextualize the output
    """

    global claim_autoais_model, claim_autoais_tokenizer
    # if autoais_model is None:
    #     logger.info("Loading AutoAIS model...")
    #     autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL, torch_dtype=torch.bfloat16, max_memory=get_max_memory(), device_map="auto")
    #     autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL, use_fast=False)

    if claim_autoais_model is None:
        logger.info("Loading Claims AutoAIS model...")
        claim_autoais_model = AutoModelForSeq2SeqLM.from_pretrained(
            OSU_AUTOAIS_MODEL,
            torch_dtype=torch.bfloat16,
            max_memory=get_max_memory(),
            device_map="auto",
        )
        # extra_special_tokens={} overrides the list stored in this model's old
        # tokenizer_config.json: transformers >= 4.53 expects a dict there and
        # otherwise crashes in _set_model_specific_special_tokens ('list' object
        # has no attribute 'keys'). The <extra_id_*> sentinels live in the
        # SentencePiece model, so dropping this kwarg is harmless for scoring.
        claim_autoais_tokenizer = AutoTokenizer.from_pretrained(
            OSU_AUTOAIS_MODEL, use_fast=False, extra_special_tokens={}
        )

    logger.info("Running AutoAIS...")

    ais_scores = []
    ais_scores_prec = []
    ais_scores_f1 = []
    ais_score_row_indices = []

    rec_supported_total = 0.0
    rec_total_total = 0
    prec_supported_total = 0.0
    prec_total_total = 0

    sent_total = 0
    sent_mcite = 0
    sent_mcite_support = 0
    sent_mcite_overcite = 0
    autoais_log = []

    for row_index, item in enumerate(tqdm(data, desc=progress_label)):
        target_sents = [item["input"] + " " + remove_citations(item["output"]).strip()]
        sents = [item["input"] + " " + item["output"]]
        citations = item["ctxs"]
        total_sents = 0

        entail = 0
        entail_prec = 0
        total_citations = 0
        total_sents = 0
        for sent_id, sent in enumerate(sents):
            # add minimum length for citation
            total_sents += 1
            target_sent = target_sents[
                sent_id
            ]  # Citation removed and (if opted for) decontextualized
            joint_entail = -1  # Undecided

            # Find references
            ref = extract_citation_indices(sent)  # In text citation id starts from 1
            # ref = [int(r[1:]) for r in re.findall(r"\[\d+", sent)]
            logger.info(f"For `{sent}`, find citations {ref}")

            if len(ref) == 0:
                # No citations
                joint_entail = 0
            elif any([ref_id >= len(citations) for ref_id in ref]):
                # Citations out of range
                joint_entail = 0
            else:
                if at_most_citations is not None:
                    ref = ref[:at_most_citations]
                total_citations += len(ref)
                joint_passage = _build_joint_passage(item["docs"], ref, "combined")

            # If not directly rejected by citation format error, calculate the recall score
            if joint_entail == -1:
                joint_entail, support_source, joint_passage = (
                    _run_autoais_with_document_variants(item["docs"], ref, target_sent)
                )
                autoais_log.append(
                    {
                        "question": item["question"],
                        "output": item["output"],
                        "claim": sent,
                        "passage": [joint_passage],
                        "support_source": support_source,
                        "model_type": "NLI",
                        "model_output": joint_entail,
                    }
                )

            entail += joint_entail
            if len(ref) > 1:
                sent_mcite += 1

            # calculate the precision score if applicable
            if joint_entail and len(ref) > 1:
                sent_mcite_support += 1
                # Precision check: did the model cite any unnecessary documents?
                for psgs_id in ref:
                    # condition A
                    nli_result, _, _ = _run_autoais_with_document_variants(
                        item["docs"], [psgs_id], target_sent
                    )

                    # condition B
                    if not nli_result:
                        subset_exclude = copy.deepcopy(ref)
                        subset_exclude.remove(psgs_id)
                        nli_result, _, _ = _run_autoais_with_document_variants(
                            item["docs"], subset_exclude, target_sent
                        )
                        if nli_result:  # psgs_id is not necessary
                            sent_mcite_overcite += 1
                        else:
                            entail_prec += 1
                    else:
                        entail_prec += 1
            else:
                entail_prec += joint_entail

        # Track totals across the whole run for the multi-citation summary.
        sent_total += total_sents
        rec_supported_total += entail
        rec_total_total += total_sents
        prec_supported_total += entail_prec
        prec_total_total += total_citations

        if total_sents > 0:
            rec_i = entail / total_sents
        else:
            rec_i = 0
        ais_scores.append(rec_i)

        prec_i = entail_prec / total_citations if total_citations > 0 else 0
        ais_scores_prec.append(prec_i)
        ais_scores_f1.append(_harmonic_mean(prec_i, rec_i))
        ais_score_row_indices.append(_autoais_source_row_index(item, row_index))

    if sent_total > 0 and sent_mcite > 0:
        sent_mcite_support_pct = (
            100 * sent_mcite_support / sent_mcite if sent_mcite > 0 else 0
        )
        sent_mcite_overcite_pct = (
            100 * sent_mcite_overcite / sent_mcite_support
            if sent_mcite_support > 0
            else 0
        )
        qprint(
            "Among all sentences, %.2f%% have multiple citations, among which %.2f%% are supported by the joint set, among which %.2f%% overcite."
            % (
                100 * sent_mcite / sent_total,
                sent_mcite_support_pct,
                sent_mcite_overcite_pct,
            )
        )

    citation_rec = _safe_div(rec_supported_total, rec_total_total)
    citation_prec = _safe_div(prec_supported_total, prec_total_total)
    citation_f1 = _harmonic_mean(citation_prec, citation_rec)

    return {
        "citation_rec": 100 * citation_rec,
        "citation_rec_all": ais_scores,
        "citation_prec": 100 * citation_prec,
        "citation_prec_all": ais_scores_prec,
        "citation_f1": 100 * citation_f1,
        "citation_f1_all": ais_scores_f1,
        "citation_row_indices_all": ais_score_row_indices,
        "citation_rec_supported_total": rec_supported_total,
        "citation_rec_total_total": rec_total_total,
        "citation_prec_supported_total": prec_supported_total,
        "citation_prec_total_total": prec_total_total,
    }


def _release_autoais_resources():
    """Release AutoAIS model/tokenizer references and clear CUDA cache."""
    global autoais_model, autoais_tokenizer
    global claim_autoais_model, claim_autoais_tokenizer

    autoais_model = None
    autoais_tokenizer = None
    claim_autoais_model = None
    claim_autoais_tokenizer = None
    _cleanup_cuda_memory()


def _merge_autoais_chunk_results(chunk_results):
    """Merge per-chunk AutoAIS outputs into the exact global aggregate."""
    merged_rec_all = []
    merged_prec_all = []
    merged_f1_all = []
    merged_row_indices_all = []
    rec_supported_total = 0.0
    rec_total_total = 0
    prec_supported_total = 0.0
    prec_total_total = 0
    cited_paper_weighted_sum = 0.0
    cited_paper_weighted_count = 0

    for chunk in chunk_results:
        merged_rec_all.extend(chunk.get("citation_rec_all", []))
        merged_prec_all.extend(chunk.get("citation_prec_all", []))
        merged_f1_all.extend(chunk.get("citation_f1_all", []))
        merged_row_indices_all.extend(chunk.get("citation_row_indices_all", []))
        rec_supported_total += chunk.get("citation_rec_supported_total", 0.0)
        rec_total_total += chunk.get("citation_rec_total_total", 0)
        prec_supported_total += chunk.get("citation_prec_supported_total", 0.0)
        prec_total_total += chunk.get("citation_prec_total_total", 0)
        chunk_weight = len(chunk.get("citation_rec_all", []))
        if "cited_paper_numbers" in chunk and chunk_weight > 0:
            cited_paper_weighted_sum += chunk["cited_paper_numbers"] * chunk_weight
            cited_paper_weighted_count += chunk_weight

    citation_rec = _safe_div(rec_supported_total, rec_total_total)
    citation_prec = _safe_div(prec_supported_total, prec_total_total)
    citation_f1 = _harmonic_mean(citation_prec, citation_rec)

    merged = {
        "citation_rec": 100 * citation_rec,
        "citation_rec_all": merged_rec_all,
        "citation_prec": 100 * citation_prec,
        "citation_prec_all": merged_prec_all,
        "citation_f1": 100 * citation_f1,
        "citation_f1_all": merged_f1_all,
        "citation_row_indices_all": merged_row_indices_all,
        "citation_rec_supported_total": rec_supported_total,
        "citation_rec_total_total": rec_total_total,
        "citation_prec_supported_total": prec_supported_total,
        "citation_prec_total_total": prec_total_total,
    }
    if cited_paper_weighted_count > 0:
        merged["cited_paper_numbers"] = (
            cited_paper_weighted_sum / cited_paper_weighted_count
        )
    return merged


def _compute_autoais_chunked(
    data,
    at_most_citations=None,
    chunk_size=0,
    use_short_form=False,
    reload_model_per_chunk=False,
):
    """Run AutoAIS over dataset chunks and merge results into one report."""
    if chunk_size is None or chunk_size <= 0 or chunk_size >= len(data):
        if use_short_form:
            return compute_autoais_short_form(
                data,
                at_most_citations=at_most_citations,
                progress_label="AutoAIS chunk 1/1",
            )
        return compute_autoais(
            data,
            at_most_citations=at_most_citations,
            progress_label="AutoAIS chunk 1/1",
        )

    chunk_results = []
    total_chunks = (len(data) + chunk_size - 1) // chunk_size
    print(
        ("[AutoAIS] Processing %d samples in %d chunks (chunk_size=%d).")
        % (len(data), total_chunks, chunk_size),
        flush=True,
    )
    running_rec_supported_total = 0.0
    running_rec_total_total = 0
    running_prec_supported_total = 0.0
    running_prec_total_total = 0

    for chunk_idx, start in enumerate(range(0, len(data), chunk_size), start=1):
        end = min(start + chunk_size, len(data))
        print(
            "[AutoAIS] Starting chunk %d/%d (items %d:%d)."
            % (chunk_idx, total_chunks, start, end),
            flush=True,
        )
        logger.warning(
            "Running AutoAIS chunk %d/%d on items [%d:%d).",
            chunk_idx,
            total_chunks,
            start,
            end,
        )
        chunk_data = data[start:end]
        if use_short_form:
            chunk_result = compute_autoais_short_form(
                chunk_data,
                at_most_citations=at_most_citations,
                progress_label="AutoAIS chunk %d/%d" % (chunk_idx, total_chunks),
            )
        else:
            chunk_result = compute_autoais(
                chunk_data,
                at_most_citations=at_most_citations,
                progress_label="AutoAIS chunk %d/%d" % (chunk_idx, total_chunks),
            )
        if all("_autoais_source_row_index" not in item for item in chunk_data):
            chunk_result["citation_row_indices_all"] = [
                start + row_index
                for row_index in chunk_result.get("citation_row_indices_all", [])
            ]
        chunk_results.append(chunk_result)
        running_rec_supported_total += chunk_result.get(
            "citation_rec_supported_total", 0.0
        )
        running_rec_total_total += chunk_result.get("citation_rec_total_total", 0)
        running_prec_supported_total += chunk_result.get(
            "citation_prec_supported_total", 0.0
        )
        running_prec_total_total += chunk_result.get("citation_prec_total_total", 0)

        running_rec = _safe_div(running_rec_supported_total, running_rec_total_total)
        running_prec = _safe_div(running_prec_supported_total, running_prec_total_total)
        running_f1 = _harmonic_mean(running_prec, running_rec)
        print(
            ("[AutoAIS] Finished chunk %d/%d. Cumulative rec=%.2f prec=%.2f f1=%.2f.")
            % (
                chunk_idx,
                total_chunks,
                100 * running_rec,
                100 * running_prec,
                100 * running_f1,
            ),
            flush=True,
        )
        logger.warning(
            (
                "Cumulative AutoAIS after chunk %d/%d: rec=%.2f, prec=%.2f, f1=%.2f "
                "(rec_num=%.0f rec_den=%d prec_num=%.0f prec_den=%d)"
            ),
            chunk_idx,
            total_chunks,
            100 * running_rec,
            100 * running_prec,
            100 * running_f1,
            running_rec_supported_total,
            running_rec_total_total,
            running_prec_supported_total,
            running_prec_total_total,
        )

        if reload_model_per_chunk and chunk_idx < total_chunks:
            logger.warning(
                "Releasing AutoAIS model after chunk %d to reduce GPU memory pressure.",
                chunk_idx,
            )
            _release_autoais_resources()

    return _merge_autoais_chunk_results(chunk_results)


def _parse_row_indices(raw_values, row_indices_file=None):
    tokens = []
    for value in raw_values or []:
        tokens.extend(re.split(r"[\s,]+", str(value).strip()))

    if row_indices_file:
        with open(row_indices_file) as f:
            tokens.extend(re.split(r"[\s,]+", f.read().strip()))

    row_indices = []
    seen = set()
    for token in tokens:
        if not token:
            continue
        row_index = int(token)
        if row_index not in seen:
            row_indices.append(row_index)
            seen.add(row_index)
    return row_indices


def _annotate_source_row_indices(data):
    for row_index, item in enumerate(data):
        if isinstance(item, dict):
            item.setdefault("_autoais_source_row_index", row_index)


def _filter_by_row_indices(data, row_indices):
    filtered = []
    data_len = len(data)
    for row_index in row_indices:
        if row_index < 0 or row_index >= data_len:
            raise IndexError(
                f"row index {row_index} out of range for prediction file with "
                f"{data_len} rows"
            )
        item = copy.deepcopy(data[row_index])
        item["_autoais_source_row_index"] = row_index
        filtered.append(item)
    return filtered


def compute_citation_coverage(data):
    prec = []
    rec = []
    f1 = []

    num_preds = []
    for item in data:
        preds = list(set(extract_citations(item["output"])))
        if "id_mapping" in item:
            qprint(item["id_mapping"])
            preds = [
                item["id_mapping"][p.split("[")[1].split("]")[0]]
                for p in preds
                if p.split("[")[1].split("]")[0] in item["id_mapping"]
            ]
        num_preds.append(len(preds))
        if "gold_ctxs" not in item:
            answers = list(set(extract_citations(item["answer"])))
        else:
            answers = list(set(item["gold_ctxs"]))
        qprint("answers: {0} preds: {1}".format(answers, preds))
        prec_i = (
            len([p for p in preds if p in answers]) / len(preds)
            if len(preds) > 0
            else 0
        )
        prec.append(prec_i)
        rec_i = (
            len([a for a in answers if a in preds]) / len(answers)
            if len(answers) > 0
            else 0
        )
        rec.append(rec_i)
        qprint(prec_i, rec_i)
        if (prec[-1] + rec[-1]) == 0:
            f1.append(0)
        else:
            f1.append(2 * prec[-1] * rec[-1] / (prec[-1] + rec[-1]))
    return {
        "num_preds": np.mean(num_preds),
        "evidence_prec": 100 * np.mean(prec),
        "evidence_rec": 100 * np.mean(rec),
        "evidence_f1": 100 * np.mean(f1),
    }


def _resolve_per_question_output_path(file_name, output_arg, total_files):
    if not output_arg:
        return None

    source_path = Path(file_name)
    default_name = source_path.name + ".autoais_per_question.json"

    if any(token in output_arg for token in ("{file}", "{name}", "{stem}")):
        return Path(
            output_arg.format(
                file=file_name,
                name=source_path.name,
                stem=source_path.stem,
            )
        )

    output_path = Path(output_arg)
    if total_files > 1 or output_arg.endswith("/") or output_path.is_dir():
        return output_path / default_name
    return output_path


def _aligned_metric_array(row_count, row_indices, values):
    aligned = [None] * row_count
    for row_index, value in zip(row_indices, values):
        if 0 <= row_index < row_count:
            aligned[row_index] = value
    return aligned


def _context_title(ctxs, ref_id):
    if not isinstance(ctxs, list) or ref_id < 0 or ref_id >= len(ctxs):
        return ""
    ctx = ctxs[ref_id]
    if isinstance(ctx, dict):
        return str(ctx.get("title", ""))
    return ""


def _build_per_question_autoais_payload(
    file_name, data, result, all_scores, eval_mode, source_row_count=None
):
    """Build row-aligned AutoAIS output without changing aggregate score files."""
    row_count = source_row_count if source_row_count is not None else len(data)
    row_indices = all_scores.get("citation_row_indices_all", [])
    rec_values = all_scores.get("citation_rec_all", [])
    prec_values = all_scores.get("citation_prec_all", [])
    f1_values = all_scores.get("citation_f1_all", [])

    # Backward-compatible fallback for any caller that lacks explicit row indices.
    if not row_indices and len(rec_values) == row_count:
        row_indices = list(range(row_count))

    rec_by_row = dict(zip(row_indices, rec_values))
    prec_by_row = dict(zip(row_indices, prec_values))
    f1_by_row = dict(zip(row_indices, f1_values))

    per_question = []
    for row_index, item in enumerate(data):
        source_row_index = _autoais_source_row_index(item, row_index)
        output = str(item.get("output", "") or "")
        ctxs = item.get("ctxs", [])
        context_count = len(ctxs) if hasattr(ctxs, "__len__") else 0
        citation_indices = extract_citation_indices(output)
        unique_citation_indices = sorted(set(citation_indices))
        out_of_range_citations = [
            ref_id
            for ref_id in citation_indices
            if ref_id < 0 or ref_id >= context_count
        ]

        rec = rec_by_row.get(source_row_index)
        prec = prec_by_row.get(source_row_index)
        f1 = f1_by_row.get(source_row_index)
        scored = source_row_index in rec_by_row

        per_question.append(
            {
                "row_index": source_row_index,
                "question": item.get("question") or item.get("input", ""),
                "input": item.get("input", ""),
                "autoais_scored": scored,
                "citation_rec": rec,
                "citation_prec": prec,
                "citation_f1": f1,
                "citation_rec_pct": None if rec is None else 100 * rec,
                "citation_prec_pct": None if prec is None else 100 * prec,
                "citation_f1_pct": None if f1 is None else 100 * f1,
                "answer_length_words": len(output.split()),
                "output_empty": len(output.strip()) == 0,
                "context_count": context_count,
                "citation_marker_count": len(citation_indices),
                "cited_source_count": len(unique_citation_indices),
                "out_of_range_citation_count": len(out_of_range_citations),
                "cited_context_titles": [
                    _context_title(ctxs, ref_id)
                    for ref_id in unique_citation_indices
                    if 0 <= ref_id < context_count
                ],
            }
        )

    return {
        "source_file": file_name,
        "score_file": file_name + ".score_post_fix",
        "eval_mode": eval_mode,
        "score_units": {
            "aggregate": "percent_0_to_100",
            "per_question_arrays": "fraction_0_to_1",
            "per_question_pct_fields": "percent_0_to_100",
        },
        "row_count": row_count,
        "evaluated_row_count": len(data),
        "scored_row_count": len(row_indices),
        "aggregate": result,
        "arrays": {
            "row_indices": row_indices,
            "citation_rec_all": rec_values,
            "citation_prec_all": prec_values,
            "citation_f1_all": f1_values,
        },
        "aligned_arrays": {
            "citation_rec_all": _aligned_metric_array(
                row_count, row_indices, rec_values
            ),
            "citation_prec_all": _aligned_metric_array(
                row_count, row_indices, prec_values
            ),
            "citation_f1_all": _aligned_metric_array(row_count, row_indices, f1_values),
        },
        "per_question": per_question,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--f",
        type=str,
        required=True,
        help="Output file. Should have field `question`, `output`, (ROUGE) `answer`, \
                        (accuracy) `qa_pairs`, (AIS) `docs`",
        nargs="+",
    )
    parser.add_argument(
        "--no_rouge", action="store_true", help="Do not evaluate ROUGE score"
    )
    parser.add_argument("--qa", action="store_true", help="Use the QA model")
    parser.add_argument("--single_qa", action="store_true", help="Use the QA model")

    parser.add_argument(
        "--citations", action="store_true", help="Evaluation with citation"
    )
    parser.add_argument(
        "--at_most_citations",
        type=int,
        default=3,
        help="At most take this many documents (mostly for precision)",
    )
    parser.add_argument("--claims_nli", action="store_true", help="Use claims for ELI5")
    parser.add_argument(
        "--evidence", action="store_true", help="Compute evidence coverage."
    )
    parser.add_argument("--match", action="store_true", help="Compute answer matching")
    parser.add_argument(
        "--use_input", action="store_true", help="Use input to compute the auto ais"
    )
    parser.add_argument("--source_data", type=str, default=None)
    parser.add_argument("--max_limit", type=int, default=None)
    parser.add_argument(
        "--row_indices",
        nargs="+",
        default=None,
        help=(
            "Optional zero-based prediction row indices to score. Accepts "
            "space- or comma-separated values. Useful for targeted case studies."
        ),
    )
    parser.add_argument(
        "--row_indices_file",
        type=str,
        default=None,
        help=(
            "Optional text file with zero-based prediction row indices, separated "
            "by commas, spaces, or newlines."
        ),
    )

    parser.add_argument(
        "--citations_short", action="store_true", help="Evaluation with citation"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=True,
        help="Suppress verbose logs and debug prints",
    )
    parser.add_argument(
        "--autoais_max_input_tokens",
        type=int,
        default=None,
        help="Optional hard cap for AutoAIS tokenizer input length. Disabled by default.",
    )
    parser.add_argument(
        "--autoais_oom_fallback_tokens",
        type=int,
        nargs="+",
        default=[2048, 1024, 768, 512],
        help="Token limits to try if a CUDA OOM happens during AutoAIS.",
    )
    parser.add_argument(
        "--disable_autoais_oom_fallback",
        action="store_true",
        help="Disable AutoAIS CUDA OOM retry logic.",
    )
    parser.add_argument(
        "--autoais_chunk_size",
        type=int,
        default=0,
        help=(
            "If >0, run citation AutoAIS in fixed-size chunks and merge results "
            "(e.g., 200)."
        ),
    )
    parser.add_argument(
        "--autoais_reload_model_per_chunk",
        action="store_true",
        help=(
            "When chunking AutoAIS, release and reload the model between chunks "
            "to free GPU memory."
        ),
    )
    parser.add_argument(
        "--per_question_output",
        type=str,
        default=None,
        help=(
            "Optional path or directory for row-aligned per-question AutoAIS JSON. "
            "For multiple --f files, pass a directory or a template containing "
            "{file}, {name}, or {stem}."
        ),
    )

    args = parser.parse_args()

    global QUIET
    global AUTOAIS_MAX_INPUT_TOKENS
    global AUTOAIS_OOM_FALLBACK_TOKEN_LIMITS
    global AUTOAIS_ENABLE_OOM_FALLBACK
    QUIET = args.quiet
    if QUIET:
        logger.setLevel(logging.ERROR)

    AUTOAIS_MAX_INPUT_TOKENS = args.autoais_max_input_tokens
    AUTOAIS_OOM_FALLBACK_TOKEN_LIMITS = sorted(
        list(set(args.autoais_oom_fallback_tokens)), reverse=True
    )
    AUTOAIS_ENABLE_OOM_FALLBACK = not args.disable_autoais_oom_fallback

    if args.source_data is not None:
        pass
        # source_data = load_jsonlines(args.source_data)
        # input2claims = {item["input"]: item["claims"] for item in source_data}

    selected_row_indices = _parse_row_indices(
        args.row_indices, row_indices_file=args.row_indices_file
    )

    for file_name in args.f:
        if file_name.endswith(".json"):
            with open(file_name) as f:
                data_with_config = json.load(f)
            data = (
                data_with_config["data"]
                if type(data_with_config) is dict
                else data_with_config
            )
        else:
            data = load_jsonlines(file_name)
        source_row_count = len(data)
        _annotate_source_row_indices(data)
        if selected_row_indices:
            qprint(
                "selected row indices: {0}".format(
                    ",".join(str(row_index) for row_index in selected_row_indices)
                )
            )
            data = _filter_by_row_indices(data, selected_row_indices)
            qprint("selected data num: {0}".format(len(data)))
        if args.max_limit is not None:
            qprint("original data num: {0}".format(len(data)))
            data = [
                item for item in data if len(item["output"].split()) < args.max_limit
            ]
            qprint("filtered data num: {0}".format(len(data)))

        if (
            args.citations is True or args.citations_short is True
        ) and "docs" not in data[0]:
            for item in data:
                item["docs"] = item["ctxs"]
                if type(item["ctxs"]) is not list:
                    item["docs"] = [
                        {"text": ctx_text[0], "title": ctx_text[1]}
                        for ctx_text in list(item["ctxs"].values())
                    ]

        if "question" not in data[0]:
            for item in data:
                item["question"] = item["input"] if "input" in item else ""

        # Truncate by newline and remove on the fly search result
        logger.warning(
            "We remove all the pre/appended space/newlines and we truncate the answer by the first newline."
        )
        logger.warning(
            "We replace any on the fly search result to standard bracket citation format."
        )
        # for i in range(len(data)):
        #     data[i]['output'] = data[i]['output'].replace("<|im_end|>", "")
        #     data[i]["output"] = data[i]["output"].replace("Here is the revised answer:\n\n", "")
        #     data[i]['output'] = data[i]['output'].replace("[OUTLINE_START]", "")
        #     data[i]['output'] = data[i]['output'].replace("[OUTLINE_END]", "")
        #     data[i]['output'] = data[i]['output'].replace("[CITATION]", "")
        #     data[i]['output'] = data[i]['output'].replace("<cit.>", "")
        #     if "answer" in data[i]:
        #         data[i]['answer'] = data[i]['answer'].replace("[OUTLINE_START]", "")
        #         data[i]['answer'] = data[i]['answer'].replace("[OUTLINE_END]", "")
        #         data[i]['answer'] = data[i]['answer'].replace("[CITATION]", "")
        #         data[i]['answer'] = data[i]['answer'].replace("<cit.>", "")

        # Remove all citations for all non-AutoAIS evaluation
        normalized_data = copy.deepcopy(data)
        for i in range(len(normalized_data)):
            normalized_data[i]["output"] = remove_citations(
                normalized_data[i]["output"]
            )

        result = {}
        all_scores = {}
        result["length"] = compute_len(normalized_data)
        if args.evidence:
            result["coverage"] = compute_citation_coverage(data)
        result["str_em"], result["str_hit"] = compute_str_em(normalized_data)

        if not args.no_rouge and "answer" in normalized_data:
            rouge_results, all_scores["rougeL"] = compute_rouge(normalized_data)
            result["rougeL"] = rouge_results["rougeL"]
            result["rouge1"] = rouge_results["rouge1"]
            result["rouge2"] = rouge_results["rouge2"]

        if args.single_qa:
            result.update(compute_single_qa(normalized_data))

        if args.match:
            result.update(compute_match(normalized_data))

        if args.citations:
            ais_results = _compute_autoais_chunked(
                data,
                at_most_citations=args.at_most_citations,
                chunk_size=args.autoais_chunk_size,
                use_short_form=False,
                reload_model_per_chunk=args.autoais_reload_model_per_chunk,
            )
            result["citation_rec"] = ais_results["citation_rec"]
            result["citation_prec"] = ais_results["citation_prec"]
            if "citation_f1" in ais_results:
                result["citation_f1"] = ais_results["citation_f1"]
            all_scores["citation_rec_all"] = ais_results["citation_rec_all"]
            all_scores["citation_prec_all"] = ais_results["citation_prec_all"]
            if "citation_f1_all" in ais_results:
                all_scores["citation_f1_all"] = ais_results["citation_f1_all"]
            all_scores["citation_row_indices_all"] = ais_results.get(
                "citation_row_indices_all", []
            )
            result["cited_paper_numbers"] = ais_results["cited_paper_numbers"]

        if args.citations_short:
            ais_results = _compute_autoais_chunked(
                data,
                at_most_citations=args.at_most_citations,
                chunk_size=args.autoais_chunk_size,
                use_short_form=True,
                reload_model_per_chunk=args.autoais_reload_model_per_chunk,
            )
            result["citation_rec"] = ais_results["citation_rec"]
            result["citation_prec"] = ais_results["citation_prec"]
            if "citation_f1" in ais_results:
                result["citation_f1"] = ais_results["citation_f1"]
            all_scores["citation_rec_all"] = ais_results["citation_rec_all"]
            all_scores["citation_prec_all"] = ais_results["citation_prec_all"]
            if "citation_f1_all" in ais_results:
                all_scores["citation_f1_all"] = ais_results["citation_f1_all"]
            all_scores["citation_row_indices_all"] = ais_results.get(
                "citation_row_indices_all", []
            )

        if args.citations or args.citations_short:
            _release_autoais_resources()

        print(result)
        with open(file_name + ".score_post_fix", "w") as f:
            json.dump(result, f, indent=4)

        per_question_output_path = _resolve_per_question_output_path(
            file_name,
            args.per_question_output,
            total_files=len(args.f),
        )
        if per_question_output_path is not None:
            if not (args.citations or args.citations_short):
                raise ValueError(
                    "--per_question_output requires --citations or --citations_short"
                )
            eval_mode = "citations_short" if args.citations_short else "citations"
            payload = _build_per_question_autoais_payload(
                file_name,
                data,
                result,
                all_scores,
                eval_mode=eval_mode,
                source_row_count=source_row_count,
            )
            per_question_output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(per_question_output_path, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"Per-question AutoAIS results: {per_question_output_path}")


if __name__ == "__main__":
    main()
