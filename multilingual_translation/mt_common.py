"""Shared, torch-free helpers for machine-translation eval and SFT data prep.

Kept free of any heavy/GPU imports (torch, generate, flash_attn) so that the
CPU-only data-prep script (scripts/prepare_sft_data.py) and the GPU eval script
(multilingual_translation/run_translation.py) can both import the SAME language
names, ALMA prompt template, and FLORES+ loader. Single source of truth: change
the template here and both the SFT training data and the eval prompt move together.
"""

from datasets import load_dataset


# FLORES+ language code ("<iso639-3>_<script>") -> English display name used in
# the prompt. Covers the high/medium/low-resource languages most commonly used
# in MMT-LLM-style studies. Unknown codes fall back to the title-cased ISO part
# (with a warning), so callers still run on any FLORES+ config.
LANG_NAMES = {
    "eng_Latn": "English",
    "deu_Latn": "German",
    "fra_Latn": "French",
    "spa_Latn": "Spanish",
    "por_Latn": "Portuguese",
    "ita_Latn": "Italian",
    "nld_Latn": "Dutch",
    "ron_Latn": "Romanian",
    "swe_Latn": "Swedish",
    "dan_Latn": "Danish",
    "nob_Latn": "Norwegian",
    "fin_Latn": "Finnish",
    "pol_Latn": "Polish",
    "ces_Latn": "Czech",
    "slk_Latn": "Slovak",
    "slv_Latn": "Slovenian",
    "hrv_Latn": "Croatian",
    "hun_Latn": "Hungarian",
    "ell_Grek": "Greek",
    "bul_Cyrl": "Bulgarian",
    "rus_Cyrl": "Russian",
    "ukr_Cyrl": "Ukrainian",
    "srp_Cyrl": "Serbian",
    "tur_Latn": "Turkish",
    "arb_Arab": "Arabic",
    "heb_Hebr": "Hebrew",
    "pes_Arab": "Persian",
    "hin_Deva": "Hindi",
    "ben_Beng": "Bengali",
    "tam_Taml": "Tamil",
    "tel_Telu": "Telugu",
    "urd_Arab": "Urdu",
    "zho_Hans": "Chinese",
    "zho_Hant": "Traditional Chinese",
    "jpn_Jpan": "Japanese",
    "kor_Hang": "Korean",
    "vie_Latn": "Vietnamese",
    "tha_Thai": "Thai",
    "ind_Latn": "Indonesian",
    "zsm_Latn": "Malay",
    "swh_Latn": "Swahili",
    "isl_Latn": "Icelandic",
    "est_Latn": "Estonian",
    "lvs_Latn": "Latvian",
    "lit_Latn": "Lithuanian",
    "cat_Latn": "Catalan",
    "eus_Latn": "Basque",
    "glg_Latn": "Galician",
}


# ALMA (arXiv:2309.11674) zero-shot instruction template. The target side is
# left open after "{tgt}: " (note the trailing space). This is the single source
# of truth for both the SFT training prompt (scripts/prepare_sft_data.py) and the
# eval prompt (run_translation.py --prompt-format alma): a model SFT'd on parallel
# data sees byte-for-byte the same prompt at eval time. Do NOT change the
# spacing/newlines without re-generating the SFT data.
ALMA_TEMPLATE = "Translate this from {src} into {tgt}:\n{src}: {src_text}\n{tgt}: "


def lang_name(code: str) -> str:
    if code in LANG_NAMES:
        return LANG_NAMES[code]
    iso = code.split("_")[0]
    name = iso.capitalize()
    print(
        f"[warn] No display name for FLORES+ code '{code}'; "
        f"using '{name}'. Add it to LANG_NAMES for a nicer prompt.",
        flush=True,
    )
    return name


def build_alma_prompt(src_text: str, src_name: str, tgt_name: str) -> str:
    """Zero-shot ALMA instruction prompt for one source sentence."""
    return ALMA_TEMPLATE.format(src=src_name, tgt=tgt_name, src_text=src_text)


def load_parallel(src_lang: str, tgt_lang: str, split: str) -> list[tuple[int, str, str]]:
    """Return (id, src_text, tgt_text) triples aligned by FLORES+ sentence id."""
    src_ds = load_dataset("openlanguagedata/flores_plus", src_lang, split=split)
    tgt_ds = load_dataset("openlanguagedata/flores_plus", tgt_lang, split=split)
    src_by_id = {int(r["id"]): r["text"] for r in src_ds}
    tgt_by_id = {int(r["id"]): r["text"] for r in tgt_ds}
    common = sorted(set(src_by_id) & set(tgt_by_id))
    return [(i, src_by_id[i], tgt_by_id[i]) for i in common]
