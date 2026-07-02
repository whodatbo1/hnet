"""Build packed SFT data for ALMA-style translation fine-tuning of H-Net.

Renders parallel sentence pairs into the ALMA instruction prompt and writes a
PACKED byte stream plus a byte-aligned label stream, ready for the `sft: true`
path in train.py (see hnet/utils/data.py:SFTByteDataset).

Output (under <data-dir>/<out-name>/):
    train.bin, val.bin               flat uint8 token stream  (BOS + prompt + target + EOS)
    labels_train.bin, labels_val.bin flat int16 label stream  (-100 on BOS+prompt, byte id on target+EOS)

Per example, with display names from mt_common.LANG_NAMES and the shared
ALMA_TEMPLATE:

    prompt = "Translate this from {src} into {tgt}:\\n{src}: {src_text}\\n{tgt}: "
    tokens = [BOS] + prompt_bytes + target_bytes + [EOS]
    labels = [-100]*(1 + len(prompt_bytes)) + target_bytes + [EOS]

so loss (cross_entropy(ignore_index=-100)) is computed only on the target
translation and the terminating EOS — exactly ALMA's "loss on target tokens
only". The training loop derives input=tokens[:-1], targets=labels[1:]; because
both streams are stored aligned and shifted identically, the first target byte
is supervised when predicted from the last prompt byte. Packed mode (mask=None)
requires equal-length rows, so we pack examples back-to-back with NO padding;
the loader chunks the flat stream into seq_len+1 windows just like pretraining.

Sources (combine freely; specify at least one):
    --pairs        FLORES+ (openlanguagedata/flores_plus) directions, via
                   mt_common.load_parallel. NOTE: train on the `dev` split and keep
                   `devtest` for eval (run_translation.py defaults to devtest) to
                   avoid leakage.
    --wmt          Classic WMT newstest sets via sacrebleu — ALMA's parallel source
                   (WMT'17-'20 by default via --wmt-sets; any year sacrebleu ships
                   works). Human-written references. WMT never covered some languages
                   (e.g. Bulgarian); missing langpairs are skipped with a warning.
    --wmt24pp      WMT24++ (google/wmt24pp): high-quality post-edited references,
                   English-source only (~960 good pairs/language after dropping
                   is_bad_source canary rows). Covers 55 languages incl. Bulgarian.
    --extra-parallel  Arbitrary "src<TAB>tgt" TSV files.
ALMA itself SFTs on WMT test sets + FLORES dev/test (58K examples total across all
languages) and finds that keeping the total SMALL beats large noisy data — so do
not also report scores on a test set you trained on.

Usage (en<->de, ALMA-style: WMT'17-'20 + FLORES dev):
    python scripts/prepare_sft_data.py \\
        --data-dir /scratch-shared/mivanov1/hnet/data --out-name sft-alma-en-de \\
        --pairs eng_Latn:deu_Latn deu_Latn:eng_Latn \\
        --wmt eng_Latn:deu_Latn deu_Latn:eng_Latn --wmt-sets wmt17 wmt18 wmt19 wmt20 \\
        --flores-split dev --val-fraction 0.05

Usage (en<->bg, where WMT lacks Bulgarian: FLORES dev + WMT24++):
    python scripts/prepare_sft_data.py \\
        --data-dir /scratch-shared/mivanov1/hnet/data --out-name sft-alma-en-bg \\
        --pairs eng_Latn:bul_Cyrl bul_Cyrl:eng_Latn \\
        --wmt24pp eng_Latn:bul_Cyrl bul_Cyrl:eng_Latn \\
        --flores-split dev --val-fraction 0.05
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "multilingual_translation"))

from mt_common import ALMA_TEMPLATE, lang_name, load_parallel  # noqa: E402

BOS = 254
EOS = 255
IGNORE_INDEX = -100

# WMT24++ (google/wmt24pp) is English-source only: every config is "en-<xx>_<REGION>"
# with `source` (English) and `target` (post-edited human reference). This maps a
# non-English FLORES+ code to its WMT24++ config so --wmt24pp can be specified with
# the same SRC:TGT FLORES codes as --pairs. Where WMT24++ offers multiple regional
# variants (fr_FR/fr_CA, pt_PT/pt_BR, es_MX, ar_EG/ar_SA, sw_KE/sw_TZ, zh_CN/zh_TW)
# a sensible default is chosen; pass --wmt24pp-config to override.
FLORES_TO_WMT24PP = {
    "deu_Latn": "en-de_DE", "fra_Latn": "en-fr_FR", "spa_Latn": "en-es_MX",
    "por_Latn": "en-pt_PT", "ita_Latn": "en-it_IT", "nld_Latn": "en-nl_NL",
    "ron_Latn": "en-ro_RO", "swe_Latn": "en-sv_SE", "dan_Latn": "en-da_DK",
    "nob_Latn": "en-no_NO", "fin_Latn": "en-fi_FI", "pol_Latn": "en-pl_PL",
    "ces_Latn": "en-cs_CZ", "slk_Latn": "en-sk_SK", "slv_Latn": "en-sl_SI",
    "hrv_Latn": "en-hr_HR", "hun_Latn": "en-hu_HU", "ell_Grek": "en-el_GR",
    "bul_Cyrl": "en-bg_BG", "rus_Cyrl": "en-ru_RU", "ukr_Cyrl": "en-uk_UA",
    "srp_Cyrl": "en-sr_RS", "tur_Latn": "en-tr_TR", "arb_Arab": "en-ar_SA",
    "heb_Hebr": "en-he_IL", "pes_Arab": "en-fa_IR", "hin_Deva": "en-hi_IN",
    "ben_Beng": "en-bn_IN", "tam_Taml": "en-ta_IN", "tel_Telu": "en-te_IN",
    "urd_Arab": "en-ur_PK", "zho_Hans": "en-zh_CN", "zho_Hant": "en-zh_TW",
    "jpn_Jpan": "en-ja_JP", "kor_Hang": "en-ko_KR", "vie_Latn": "en-vi_VN",
    "tha_Thai": "en-th_TH", "ind_Latn": "en-id_ID", "isl_Latn": "en-is_IS",
    "est_Latn": "en-et_EE", "lvs_Latn": "en-lv_LV", "lit_Latn": "en-lt_LT",
    "cat_Latn": "en-ca_ES", "swh_Latn": "en-sw_KE",
}


def load_wmt24pp(src_flores: str, tgt_flores: str, config: str | None = None):
    """Yield (src_text, tgt_text) pairs for one direction from WMT24++.

    WMT24++ is English-source only, so exactly one of src/tgt must be eng_Latn.
    The non-English side selects the HF config (auto from FLORES_TO_WMT24PP unless
    `config` is given). is_bad_source rows (incl. contamination "canary" segments)
    are dropped. Direction is honoured: eng->xx emits (source, target); xx->eng
    emits (target, source).
    """
    from datasets import load_dataset

    if (src_flores == "eng_Latn") == (tgt_flores == "eng_Latn"):
        raise SystemExit(
            f"WMT24++ supports only English<->X pairs; got {src_flores}:{tgt_flores}. "
            f"Exactly one side must be eng_Latn."
        )
    non_en = tgt_flores if src_flores == "eng_Latn" else src_flores
    cfg = config or FLORES_TO_WMT24PP.get(non_en)
    if cfg is None:
        raise SystemExit(
            f"No WMT24++ config known for FLORES code '{non_en}'. Pass --wmt24pp-config "
            f"<en-xx_REGION> explicitly (see https://huggingface.co/datasets/google/wmt24pp)."
        )

    ds = load_dataset("google/wmt24pp", cfg, split="train")
    src_is_en = src_flores == "eng_Latn"
    n_bad = 0
    for r in ds:
        if r["is_bad_source"]:
            n_bad += 1
            continue
        english, foreign = r["source"], r["target"]
        if not (english.strip() and foreign.strip()):
            continue
        yield (english, foreign) if src_is_en else (foreign, english)
    print(f"    (wmt24pp {cfg}: dropped {n_bad} is_bad_source rows)", flush=True)


# FLORES+ code -> classic WMT (sacrebleu) ISO-639-1 language code. WMT newstest
# sets use 2-letter codes. Not every language is in every year, and several
# (e.g. Bulgarian) never appeared in WMT at all — load_wmt skips missing pairs.
FLORES_TO_WMT = {
    "eng_Latn": "en", "deu_Latn": "de", "fra_Latn": "fr", "spa_Latn": "es",
    "por_Latn": "pt", "ita_Latn": "it", "nld_Latn": "nl", "ron_Latn": "ro",
    "swe_Latn": "sv", "dan_Latn": "da", "nob_Latn": "no", "fin_Latn": "fi",
    "pol_Latn": "pl", "ces_Latn": "cs", "slk_Latn": "sk", "slv_Latn": "sl",
    "hrv_Latn": "hr", "hun_Latn": "hu", "ell_Grek": "el", "bul_Cyrl": "bg",
    "rus_Cyrl": "ru", "ukr_Cyrl": "uk", "srp_Cyrl": "sr", "tur_Latn": "tr",
    "arb_Arab": "ar", "heb_Hebr": "he", "pes_Arab": "fa", "hin_Deva": "hi",
    "ben_Beng": "bn", "tam_Taml": "ta", "tel_Telu": "te", "urd_Arab": "ur",
    "zho_Hans": "zh", "zho_Hant": "zh", "jpn_Jpan": "ja", "kor_Hang": "ko",
    "vie_Latn": "vi", "tha_Thai": "th", "ind_Latn": "id", "isl_Latn": "is",
    "est_Latn": "et", "lvs_Latn": "lv", "lit_Latn": "lt", "cat_Latn": "ca",
    "eus_Latn": "eu", "glg_Latn": "gl",
}


def load_wmt(testset: str, src_flores: str, tgt_flores: str):
    """Yield (src_text, tgt_text) pairs for one direction from a classic WMT
    testset (newstest etc.) via sacrebleu — the human-written reference sets ALMA
    uses (WMT'17-'20 by default; any year sacrebleu ships works).

    Tries the requested langpair "s-t"; if absent, falls back to the reverse "t-s"
    and swaps source/reference. Skips (with a warning) testsets lacking the pair.
    Uses the first reference when multiple are present.
    """
    import sacrebleu

    if testset not in sacrebleu.DATASETS:
        raise SystemExit(
            f"Unknown WMT testset '{testset}'. Run "
            f"`python -c 'import sacrebleu; print(sacrebleu.get_available_testsets())'`."
        )
    s = FLORES_TO_WMT.get(src_flores)
    t = FLORES_TO_WMT.get(tgt_flores)
    if s is None or t is None:
        missing = src_flores if s is None else tgt_flores
        raise SystemExit(f"No WMT language code known for FLORES code '{missing}'.")

    ds = sacrebleu.DATASETS[testset]
    langpairs = ds.langpairs
    if f"{s}-{t}" in langpairs:
        lp, swap = f"{s}-{t}", False
    elif f"{t}-{s}" in langpairs:
        lp, swap = f"{t}-{s}", True
    else:
        print(f"    [warn] {testset} has no {s}-{t} or {t}-{s}; skipping.", flush=True)
        return

    sources = list(ds.source(lp))
    references = list(ds.references(lp))
    for src_line, ref_set in zip(sources, references):
        ref_line = ref_set[0] if ref_set else ""
        src_text, tgt_text = (ref_line, src_line) if swap else (src_line, ref_line)
        if src_text.strip() and tgt_text.strip():
            yield src_text, tgt_text


def render_example(src_text: str, tgt_text: str, src_name: str, tgt_name: str):
    """Return (tokens uint8 list, labels int list) for one parallel pair.

    labels[j] == tokens[j] for the target region (translation + EOS), else -100.
    """
    prompt = ALMA_TEMPLATE.format(src=src_name, tgt=tgt_name, src_text=src_text)
    prompt_bytes = list(prompt.encode("utf-8"))
    target_bytes = list(tgt_text.encode("utf-8"))

    tokens = [BOS] + prompt_bytes + target_bytes + [EOS]
    labels = (
        [IGNORE_INDEX] * (1 + len(prompt_bytes))  # BOS + prompt: not supervised
        + target_bytes                            # translation: supervised
        + [EOS]                                   # learn to stop
    )
    assert len(tokens) == len(labels)
    return tokens, labels


def gather_examples(args) -> list[tuple[str, str, str, str]]:
    """Collect (src_text, tgt_text, src_name, tgt_name) tuples from all sources."""
    examples: list[tuple[str, str, str, str]] = []

    for pair in args.pairs:
        if ":" not in pair:
            raise SystemExit(f"--pairs entry '{pair}' must be SRC:TGT (e.g. eng_Latn:bul_Cyrl).")
        src_lang, tgt_lang = pair.split(":", 1)
        src_name, tgt_name = lang_name(src_lang), lang_name(tgt_lang)
        n_before = len(examples)
        for split in args.flores_split:
            rows = load_parallel(src_lang, tgt_lang, split)
            for _, src_text, tgt_text in rows:
                if src_text.strip() and tgt_text.strip():
                    examples.append((src_text, tgt_text, src_name, tgt_name))
        print(f"  {src_lang}->{tgt_lang}: +{len(examples) - n_before} pairs "
              f"(splits={args.flores_split})", flush=True)

    for pair in args.wmt or []:
        if ":" not in pair:
            raise SystemExit(f"--wmt entry '{pair}' must be SRC:TGT (e.g. eng_Latn:deu_Latn).")
        src_lang, tgt_lang = pair.split(":", 1)
        src_name, tgt_name = lang_name(src_lang), lang_name(tgt_lang)
        for testset in args.wmt_sets:
            n_before = len(examples)
            for src_text, tgt_text in load_wmt(testset, src_lang, tgt_lang):
                examples.append((src_text, tgt_text, src_name, tgt_name))
            added = len(examples) - n_before
            if added:
                print(f"  wmt {testset} {src_lang}->{tgt_lang}: +{added} pairs", flush=True)

    for pair in args.wmt24pp or []:
        if ":" not in pair:
            raise SystemExit(f"--wmt24pp entry '{pair}' must be SRC:TGT (e.g. eng_Latn:bul_Cyrl).")
        src_lang, tgt_lang = pair.split(":", 1)
        src_name, tgt_name = lang_name(src_lang), lang_name(tgt_lang)
        n_before = len(examples)
        for src_text, tgt_text in load_wmt24pp(src_lang, tgt_lang, args.wmt24pp_config):
            examples.append((src_text, tgt_text, src_name, tgt_name))
        print(f"  wmt24pp {src_lang}->{tgt_lang}: +{len(examples) - n_before} pairs", flush=True)

    for spec in args.extra_parallel or []:
        path, src_lang, tgt_lang = spec
        src_name, tgt_name = lang_name(src_lang), lang_name(tgt_lang)
        n_before = len(examples)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if "\t" not in line:
                    continue
                src_text, tgt_text = line.split("\t", 1)
                if src_text.strip() and tgt_text.strip():
                    examples.append((src_text, tgt_text, src_name, tgt_name))
        print(f"  extra {path} ({src_lang}->{tgt_lang}): +{len(examples) - n_before} pairs",
              flush=True)

    return examples


def pack(examples: list[tuple[str, str, str, str]]):
    """Render and concatenate examples into flat token/label arrays."""
    tok_parts: list[np.ndarray] = []
    lab_parts: list[np.ndarray] = []
    for src_text, tgt_text, src_name, tgt_name in examples:
        tokens, labels = render_example(src_text, tgt_text, src_name, tgt_name)
        tok_parts.append(np.asarray(tokens, dtype=np.uint8))
        lab_parts.append(np.asarray(labels, dtype=np.int16))
    if not tok_parts:
        return np.empty(0, dtype=np.uint8), np.empty(0, dtype=np.int16)
    return np.concatenate(tok_parts), np.concatenate(lab_parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", required=True,
                        help="Root data dir (same one train.py reads via data_dir).")
    parser.add_argument("--out-name", required=True,
                        help="Subdir under --data-dir to write the .bin files into "
                             "(e.g. sft-alma-en-bg). Use this as `sft_name` in the SFT config.")
    parser.add_argument("--pairs", nargs="*", default=[],
                        help="FLORES+ directions SRC:TGT (list both ways for bidirectional), "
                             "e.g. eng_Latn:bul_Cyrl bul_Cyrl:eng_Latn. Sourced from --flores-split.")
    parser.add_argument("--flores-split", nargs="+", default=["dev"],
                        choices=["dev", "devtest"],
                        help="FLORES+ split(s) to source --pairs from. Keep 'devtest' "
                             "OUT of training if you eval on devtest (default: dev only).")
    parser.add_argument("--wmt", nargs="*", default=[],
                        help="Classic WMT newstest directions SRC:TGT (via sacrebleu), e.g. "
                             "eng_Latn:deu_Latn deu_Latn:eng_Latn. Pulled from each year in "
                             "--wmt-sets. This is ALMA's parallel source (WMT'17-'20). NOTE: "
                             "WMT never covered some languages (e.g. Bulgarian) — those are skipped.")
    parser.add_argument("--wmt-sets", nargs="+",
                        default=["wmt17", "wmt18", "wmt19", "wmt20"],
                        help="WMT testset names for --wmt (default: ALMA's wmt17-wmt20). Any year "
                             "sacrebleu ships works, e.g. wmt21 wmt22 wmt23 wmt24.")
    parser.add_argument("--wmt24pp", nargs="*", default=[],
                        help="WMT24++ (google/wmt24pp) directions SRC:TGT, e.g. "
                             "eng_Latn:bul_Cyrl bul_Cyrl:eng_Latn. English-source only; the "
                             "non-English code selects the config via FLORES_TO_WMT24PP. "
                             "is_bad_source rows are dropped. Covers 55 languages incl. Bulgarian.")
    parser.add_argument("--wmt24pp-config", default=None,
                        help="Override the auto-selected WMT24++ config (e.g. en-pt_BR) for all "
                             "--wmt24pp entries. Use only when doing a single language.")
    parser.add_argument("--extra-parallel", nargs=3, action="append",
                        metavar=("TSV", "SRC", "TGT"), default=None,
                        help="Optional extra parallel data: a TSV of 'src<TAB>tgt' lines "
                             "plus its SRC and TGT FLORES+ codes. Repeatable.")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Gathering parallel examples...", flush=True)
    examples = gather_examples(args)
    if not examples:
        raise SystemExit("No examples gathered; specify at least one of "
                         "--pairs / --wmt24pp / --extra-parallel.")
    print(f"Total examples: {len(examples)}", flush=True)

    # Shuffle the EXAMPLE list (not bytes) so packing interleaves directions/sources,
    # then split into train/val before packing (never split mid-example).
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(examples))
    n_val = int(len(examples) * args.val_fraction)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    train_examples = [examples[i] for i in train_idx]
    val_examples = [examples[i] for i in val_idx]
    print(f"Split: {len(train_examples)} train / {len(val_examples)} val", flush=True)

    out_dir = Path(args.data_dir) / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, exs in (("train", train_examples), ("val", val_examples)):
        if not exs:
            print(f"[warn] no {split_name} examples; skipping (val-fraction too small?).",
                  flush=True)
            continue
        tokens, labels = pack(exs)
        tok_path = out_dir / f"{split_name}.bin"
        lab_path = out_dir / f"labels_{split_name}.bin"
        tokens.tofile(tok_path)
        labels.tofile(lab_path)
        n_supervised = int((labels != IGNORE_INDEX).sum())
        print(f"  {split_name}: {len(exs)} examples, {tokens.size} tokens "
              f"({n_supervised} supervised, {100*n_supervised/max(tokens.size,1):.1f}%)\n"
              f"    -> {tok_path} ({tokens.nbytes} B uint8)\n"
              f"    -> {lab_path} ({labels.nbytes} B int16)", flush=True)

    print("\nDone. Set in your SFT config:")
    print(f"  sft: true")
    print(f"  sft_name: {args.out_name}")
    print(f"  data_dir: {args.data_dir}")


if __name__ == "__main__":
    main()
