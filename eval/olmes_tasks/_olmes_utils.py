"""Shared process_docs helpers for OLMES task YAMLs.

These mirror what `oe_eval/tasks/oe_eval_tasks/{hellaswag,winogrande,...}.py`
do at evaluation time so the in-context-example formatting in lm-eval matches
the OLMES specification.
"""
import re

import datasets


# ---------- HellaSwag -----------------------------------------------------

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


def hellaswag_mcf_doc_to_text(doc):
    endings = [_hellaswag_clean(e).lstrip() for e in doc["endings"]]
    labels = ["A", "B", "C", "D"][: len(endings)]
    body = "\n".join(f" {lab}. {text}" for lab, text in zip(labels, endings))
    return f"{_hellaswag_query(doc)}\n{body}\nAnswer:"


def hellaswag_mcf_doc_to_choice(doc):
    return ["A", "B", "C", "D"][: len(doc["endings"])]


def hellaswag_mcf_doc_to_target(doc):
    return int(doc["label"])


# ---------- Winogrande ----------------------------------------------------

def winogrande_cf_doc_to_text(doc):
    """CF: returns the gold-choice integer index; the actual scored sequences
    come from `doc_to_choice` (each is a full sentence with one option spliced
    in, and the continuation is the suffix after `_`).
    """
    return {"1": 0, "2": 1}[doc["answer"]]


def winogrande_cf_doc_to_target(doc):
    idx = doc["sentence"].index("_") + 1
    return doc["sentence"][idx:].strip()


def winogrande_cf_doc_to_choice(doc):
    idx = doc["sentence"].index("_")
    return [doc["sentence"][:idx] + opt for opt in (doc["option1"], doc["option2"])]


def winogrande_mcf_doc_to_text(doc):
    return (
        f"Fill in the blank: {doc['sentence']}\n"
        f" A. {doc['option1']}\n"
        f" B. {doc['option2']}\n"
        f"Answer:"
    )


def winogrande_mcf_doc_to_choice(doc):
    return ["A", "B"]


def winogrande_mcf_doc_to_target(doc):
    return {"1": 0, "2": 1}[doc["answer"]]


# ---------- Generic MCF helpers ------------------------------------------
# OLMES `make_mcq_prompt` puts a leading space before each label
# (` A. <text>`); Jinja's lstrip_blocks behavior in lm-eval strips that, so
# we render the choice block in Python instead.

def _olmes_mc_block(question, choices_text, choices_label):
    body = "\n".join(
        f" {lab}. {text}" for lab, text in zip(choices_label, choices_text)
    )
    return f"Question: {question}\n{body}\nAnswer:"


def arc_mcf_doc_to_text(doc):
    return _olmes_mc_block(
        doc["question"], doc["choices"]["text"], doc["choices"]["label"]
    )


def arc_mcf_doc_to_choice(doc):
    return list(doc["choices"]["label"])


def arc_mcf_doc_to_target(doc):
    return doc["choices"]["label"].index(doc["answerKey"])


def csqa_mcf_doc_to_text(doc):
    return _olmes_mc_block(
        doc["question"], doc["choices"]["text"], doc["choices"]["label"]
    )


def csqa_mcf_doc_to_choice(doc):
    return list(doc["choices"]["label"])


def csqa_mcf_doc_to_target(doc):
    return doc["choices"]["label"].index(doc["answerKey"])


def openbookqa_mcf_doc_to_text(doc):
    return _olmes_mc_block(
        doc["question_stem"], doc["choices"]["text"], doc["choices"]["label"]
    )


def openbookqa_mcf_doc_to_choice(doc):
    return list(doc["choices"]["label"])


def openbookqa_mcf_doc_to_target(doc):
    return doc["choices"]["label"].index(doc["answerKey"].strip())
