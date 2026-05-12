"""Doc-processing helpers for EU20 task YAMLs.

Mirrors what `eval/olmes_tasks/_olmes_utils.py` does but in a language-aware
way: the MCF prompts ("Question:", "Answer:") and the TruthfulQA prompt are
localized per language. GSM8K answer extraction follows the upstream
`#### N` convention preserved by the EU20 translations.
"""
import re

# ----- localized prompt fragments ------------------------------------------
# Keys are the EU20 language codes used in dataset filenames and yaml task
# names. EN reuses the upstream English benchmarks via this same module.

QA_LABELS = {
    "EN": ("Question", "Answer"),
    "NL": ("Vraag", "Antwoord"),
    "FI": ("Kysymys", "Vastaus"),
    "BG": ("Въпрос", "Отговор"),
}


# ----- ARC ----------------------------------------------------------------
# CF templates are inline strings in the yaml; only MCF needs Python because
# the choice-letter rendering (` A. <text>`) gets stripped by Jinja's
# lstrip_blocks. (Same reason as _olmes_utils._olmes_mc_block.)

def _arc_mcf_text(doc, q_label, a_label):
    body = "\n".join(
        f" {lab}. {text}"
        for lab, text in zip(doc["choices"]["label"], doc["choices"]["text"])
    )
    return f"{q_label}: {doc['question']}\n{body}\n{a_label}:"


def arc_mcf_doc_to_text_EN(doc): return _arc_mcf_text(doc, *QA_LABELS["EN"])
def arc_mcf_doc_to_text_NL(doc): return _arc_mcf_text(doc, *QA_LABELS["NL"])
def arc_mcf_doc_to_text_FI(doc): return _arc_mcf_text(doc, *QA_LABELS["FI"])
def arc_mcf_doc_to_text_BG(doc): return _arc_mcf_text(doc, *QA_LABELS["BG"])


def arc_mcf_doc_to_choice(doc):
    return list(doc["choices"]["label"])


def arc_mcf_doc_to_target(doc):
    return doc["choices"]["label"].index(doc["answerKey"])


# ----- HellaSwag ----------------------------------------------------------
# Assumes Eurolingua/hellaswagx preserves the upstream Rowan/hellaswag fields
# (ctx_a, ctx_b, activity_label, endings, label). If a future inspection
# shows pre-concatenated `ctx`, swap `_hellaswag_query` accordingly.

def _hellaswag_clean(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\.? \[title\]", ". ", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def _hellaswag_query(doc):
    ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
    return _hellaswag_clean(doc["activity_label"] + ": " + ctx)


def hellaswag_cf_doc_to_text(doc):
    return _hellaswag_query(doc)


def hellaswag_cf_doc_to_choice(doc):
    return [_hellaswag_clean(e) for e in doc["endings"]]


def hellaswag_cf_doc_to_target(doc):
    return int(doc["label"])


def _hellaswag_mcf_text(doc, a_label):
    endings = [_hellaswag_clean(e).lstrip() for e in doc["endings"]]
    labels = ["A", "B", "C", "D"][: len(endings)]
    body = "\n".join(f" {lab}. {text}" for lab, text in zip(labels, endings))
    return f"{_hellaswag_query(doc)}\n{body}\n{a_label}:"


def hellaswag_mcf_doc_to_text_EN(doc): return _hellaswag_mcf_text(doc, QA_LABELS["EN"][1])
def hellaswag_mcf_doc_to_text_NL(doc): return _hellaswag_mcf_text(doc, QA_LABELS["NL"][1])
def hellaswag_mcf_doc_to_text_FI(doc): return _hellaswag_mcf_text(doc, QA_LABELS["FI"][1])
def hellaswag_mcf_doc_to_text_BG(doc): return _hellaswag_mcf_text(doc, QA_LABELS["BG"][1])


def hellaswag_mcf_doc_to_choice(doc):
    return ["A", "B", "C", "D"][: len(doc["endings"])]


def hellaswag_mcf_doc_to_target(doc):
    return int(doc["label"])


# ----- TruthfulQA ---------------------------------------------------------
# The Eurolingua/truthfulqax mc files keep the upstream truthful_qa schema:
# {question, mc1_targets: {choices, labels}, mc2_targets: {choices, labels}, id}
# Prompt convention: "Q: ...\nA:" per the original lm-eval truthfulqa_mc1
# task. Localized labels per language.

def _tqa_text(doc, q_label, a_label):
    short_q = "Q" if q_label == "Question" else q_label[0]  # keep short like "V" for NL
    short_a = "A" if a_label == "Answer" else a_label[0]
    return f"{short_q}: {doc['question']}\n{short_a}:"


def tqa_doc_to_text_EN(doc): return _tqa_text(doc, *QA_LABELS["EN"])
def tqa_doc_to_text_NL(doc): return _tqa_text(doc, *QA_LABELS["NL"])
def tqa_doc_to_text_FI(doc): return _tqa_text(doc, *QA_LABELS["FI"])
def tqa_doc_to_text_BG(doc): return _tqa_text(doc, *QA_LABELS["BG"])


def tqa_mc1_doc_to_choice(doc):
    return list(doc["mc1_targets"]["choices"])


def tqa_mc1_doc_to_target(doc):
    return list(doc["mc1_targets"]["labels"]).index(1)


def tqa_mc2_doc_to_choice(doc):
    return list(doc["mc2_targets"]["choices"])


def tqa_mc2_doc_to_target(doc):
    # multi-correct; the metric uses sum-of-correct-prob (process_results below)
    return list(doc["mc2_targets"]["labels"]).index(1)


def tqa_mc2_process_results(doc, results):
    """Sum of softmax probability over the correct continuations.

    `results` is a list of (loglikelihood, is_greedy) per choice — same shape
    lm-eval passes for multiple_choice tasks.
    """
    import math
    lls = [r[0] for r in results]
    labels = list(doc["mc2_targets"]["labels"])
    m = max(lls)
    exps = [math.exp(ll - m) for ll in lls]
    z = sum(exps)
    probs = [e / z for e in exps]
    correct = sum(p for p, lab in zip(probs, labels) if lab == 1)
    return {"acc": float(correct)}


# MCF for TruthfulQA mc1 — render A./B./C./D. and score the gold letter.

def _tqa_mc1_mcf_text(doc, q_label, a_label):
    short_q = "Q" if q_label == "Question" else q_label[0]
    short_a = "A" if a_label == "Answer" else a_label[0]
    choices = list(doc["mc1_targets"]["choices"])
    labels = ["A", "B", "C", "D"][: len(choices)]
    body = "\n".join(f" {lab}. {text}" for lab, text in zip(labels, choices))
    return f"{short_q}: {doc['question']}\n{body}\n{short_a}:"


def tqa_mc1_mcf_doc_to_text_EN(doc): return _tqa_mc1_mcf_text(doc, *QA_LABELS["EN"])
def tqa_mc1_mcf_doc_to_text_NL(doc): return _tqa_mc1_mcf_text(doc, *QA_LABELS["NL"])
def tqa_mc1_mcf_doc_to_text_FI(doc): return _tqa_mc1_mcf_text(doc, *QA_LABELS["FI"])
def tqa_mc1_mcf_doc_to_text_BG(doc): return _tqa_mc1_mcf_text(doc, *QA_LABELS["BG"])


def tqa_mc1_mcf_doc_to_choice(doc):
    return ["A", "B", "C", "D"][: len(doc["mc1_targets"]["choices"])]


def tqa_mc1_mcf_doc_to_target(doc):
    return list(doc["mc1_targets"]["labels"]).index(1)


# ----- GSM8K --------------------------------------------------------------
# EU20 preserves the upstream "#### N" final-answer marker in `answer`.

_GSM8K_ANS_RE = re.compile(r"####\s*([\-0-9.,]+)")


def _gsm8k_extract(text):
    m = _GSM8K_ANS_RE.search(text or "")
    if not m:
        return None
    return m.group(1).replace(",", "").strip().rstrip(".")


def gsm8k_doc_to_text_EN(doc): return f"Question: {doc['question']}\nAnswer:"
def gsm8k_doc_to_text_NL(doc): return f"Vraag: {doc['question']}\nAntwoord:"
def gsm8k_doc_to_text_FI(doc): return f"Kysymys: {doc['question']}\nVastaus:"
def gsm8k_doc_to_text_BG(doc): return f"Въпрос: {doc['question']}\nОтговор:"


def gsm8k_doc_to_target(doc):
    """Few-shot target: the full chain-of-thought answer (with #### N)."""
    return " " + doc["answer"]


def gsm8k_process_results(doc, results):
    """Exact-match between the numbers extracted from generated text and gold."""
    gold = _gsm8k_extract(doc["answer"])
    pred_text = results[0] if results else ""
    pred = _gsm8k_extract(pred_text) or _last_number(pred_text)
    return {"exact_match": float(gold is not None and pred is not None and gold == pred)}


_NUM_RE = re.compile(r"-?\d+(?:[\.,]\d+)?")


def _last_number(text):
    if not text:
        return None
    matches = _NUM_RE.findall(text)
    if not matches:
        return None
    return matches[-1].replace(",", "")
