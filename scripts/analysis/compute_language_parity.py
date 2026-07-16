#!/usr/bin/env python3
"""Compute byte and tokenizer parity of languages vs English on FLORES-200.

FLORES-200 is sentence-aligned across all languages, so for every language
the same set of sentences expresses the same meaning. Parity of language X
is then the ratio of the cost of encoding X to the cost of encoding the
English side:

    byte parity      = utf8_bytes(X) / utf8_bytes(eng_Latn)
    tokenizer parity = tokens(X)     / tokens(eng_Latn)

Tokenizer parity is computed for the GPT-2 and Llama-3 tokenizers.

The FLORES-200 dataset is downloaded once from Meta's public mirror
(https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz, ~25 MB)
and cached under <data-dir>/flores200_dataset.

Usage (needs the `hnet` env for transformers + HF token for Llama-3):
    python scripts/analysis/compute_language_parity.py
    python scripts/analysis/compute_language_parity.py \\
        --languages bul_Cyrl nld_Latn kor_Hang cmn_Hans \\
        --split devtest \\
        --output language_parity.json
"""

import argparse
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
ENGLISH = "eng_Latn"

# Languages used across this project's training/eval runs (FLORES-200 codes).
DEFAULT_LANGUAGES = [
    "bul_Cyrl",
    "nld_Latn",
    "deu_Latn",
    "fra_Latn",
    "fin_Latn",
    "kor_Hang",
    "cmn_Hans",
    "arb_Arab",
]

# HPLT-style codes used elsewhere in this repo -> FLORES-200 file names.
FLORES_ALIASES = {"cmn_Hans": "zho_Hans", "cmn_Hant": "zho_Hant"}

TOKENIZERS = {
    "gpt2": "openai-community/gpt2",
    "llama3": "meta-llama/Meta-Llama-3-8B",
}


def download_flores(data_dir: Path) -> Path:
    """Download and extract FLORES-200 into <data_dir>/flores200_dataset."""
    dataset_dir = data_dir / "flores200_dataset"
    if (dataset_dir / "devtest" / f"{ENGLISH}.devtest").exists():
        print(f"FLORES-200 cache found at {dataset_dir}")
        return dataset_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    tar_path = data_dir / "flores200_dataset.tar.gz"
    print(f"Downloading {FLORES_URL} ...")
    urllib.request.urlretrieve(FLORES_URL, tar_path)
    print(f"Extracting to {dataset_dir} ...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(data_dir)
    tar_path.unlink()
    return dataset_dir


def load_sentences(dataset_dir: Path, lang: str, splits: list[str]) -> list[str]:
    flores_code = FLORES_ALIASES.get(lang, lang)
    sentences = []
    for split in splits:
        path = dataset_dir / split / f"{flores_code}.{split}"
        if not path.exists():
            sys.exit(f"error: {path} not found — is '{lang}' a valid FLORES-200 code?")
        sentences.extend(line for line in path.read_text(encoding="utf-8").splitlines() if line)
    return sentences


def main():
    parser = argparse.ArgumentParser(
        description="Byte and tokenizer parity vs English on FLORES-200"
    )
    parser.add_argument("--languages", nargs="+", default=DEFAULT_LANGUAGES,
                        help="FLORES-200 language codes to compare against English "
                             f"(default: {' '.join(DEFAULT_LANGUAGES)})")
    parser.add_argument("--split", choices=["dev", "devtest", "all"], default="all",
                        help="FLORES-200 split(s) to use (default: all = dev + devtest)")
    parser.add_argument("--data-dir", type=Path,
                        default=Path("/projects/0/hpmlprjs/interns/marko/hnet/data"),
                        help="Directory holding/receiving the FLORES-200 download")
    parser.add_argument("--output", type=Path, default=Path("language_parity.json"),
                        help="Output JSON file (default: language_parity.json)")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    splits = ["dev", "devtest"] if args.split == "all" else [args.split]
    dataset_dir = download_flores(args.data_dir)

    print("Loading tokenizers ...")
    tokenizers = {name: AutoTokenizer.from_pretrained(repo)
                  for name, repo in TOKENIZERS.items()}

    languages = [ENGLISH] + [l for l in args.languages if l != ENGLISH]
    totals = {}
    for lang in languages:
        sentences = load_sentences(dataset_dir, lang, splits)
        byte_count = sum(len(s.encode("utf-8")) for s in sentences)
        counts = {"n_sentences": len(sentences), "bytes": byte_count}
        for name, tok in tokenizers.items():
            encoded = tok(sentences, add_special_tokens=False)["input_ids"]
            counts[f"{name}_tokens"] = sum(len(ids) for ids in encoded)
        totals[lang] = counts
        print(f"  {lang}: {counts['n_sentences']} sentences, "
              f"{counts['bytes']:,} bytes, "
              f"{counts['gpt2_tokens']:,} gpt2 tokens, "
              f"{counts['llama3_tokens']:,} llama3 tokens")

    eng = totals[ENGLISH]
    n_expected = eng["n_sentences"]
    for lang, counts in totals.items():
        if counts["n_sentences"] != n_expected:
            sys.exit(f"error: {lang} has {counts['n_sentences']} sentences, "
                     f"expected {n_expected} — splits are not aligned")

    results = {
        "dataset": "FLORES-200",
        "splits": splits,
        "n_sentences": n_expected,
        "baseline": ENGLISH,
        "tokenizers": TOKENIZERS,
        "languages": {
            lang: {
                **counts,
                "byte_parity": counts["bytes"] / eng["bytes"],
                "gpt2_parity": counts["gpt2_tokens"] / eng["gpt2_tokens"],
                "llama3_parity": counts["llama3_tokens"] / eng["llama3_tokens"],
            }
            for lang, counts in totals.items()
        },
    }

    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"\nWrote {args.output}")

    print(f"\n{'language':<12} {'byte parity':>12} {'gpt2 parity':>12} {'llama3 parity':>14}")
    for lang, r in results["languages"].items():
        print(f"{lang:<12} {r['byte_parity']:>12.3f} {r['gpt2_parity']:>12.3f} "
              f"{r['llama3_parity']:>14.3f}")


if __name__ == "__main__":
    main()
