"""Per-sentence patch boundaries on FLORES-200 for one checkpoint + language.

FLORES-200 is sentence-aligned across languages: line i of <lang>.<split>
translates line i of eng_Latn.<split>. Every record in the output is keyed by
that line index (``id``), so results from different (checkpoint, language)
runs can be joined per sentence — e.g. compare how model A segments English
sentence 42 vs how model B segments its Bulgarian translation.

The boundary computation reuses compute_boundaries from
scripts/visualize_boundaries.py (router-agnostic, full forward pass), and the
FLORES-200 download/cache from scripts/analysis/compute_language_parity.py.
BOS is excluded from all counts/positions (same convention as
scripts/compression_flores.py), so metrics reflect the routing policy on
actual text bytes.

Usage:
    # Compute (needs a GPU node)
    python scripts/flores_boundaries.py run \\
        --model-path /path/to/ckpt/model_step40000.pt \\
        --config-path configs/comparison/XXS/hnet_1stage_XXS.json \\
        --lang bul_Cyrl \\
        --split devtest \\
        --output boundaries_en_nl_model_bul.json

    # Compare two result files on the same sentence(s)
    python scripts/flores_boundaries.py compare a.json b.json --id 42 17
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))          # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent / "analysis"))

from compute_language_parity import FLORES_ALIASES, download_flores


def load_flores_split(dataset_dir: Path, lang: str, split: str):
    """Return [(line_idx, sentence), ...] for one language/split.

    line_idx is the 0-based line number in the FLORES-200 file — the
    cross-language sentence id. Empty lines (none in practice) are skipped
    without shifting the ids of later sentences.
    """
    flores_code = FLORES_ALIASES.get(lang, lang)
    path = dataset_dir / split / f"{flores_code}.{split}"
    if not path.exists():
        sys.exit(f"error: {path} not found — is '{lang}' a valid FLORES-200 code?")
    lines = path.read_text(encoding="utf-8").splitlines()
    return [(i, s) for i, s in enumerate(lines) if s]


def boundary_quality(prob, mask):
    """Router-quality stats for one sentence (same definitions as
    experiments/boundary_metrics.py): mean binary entropy H_b of the boundary
    probabilities in bits (low = decisive router), and mean probability at
    selected (G_pos) / non-selected (G_neg) positions — the separation margin.

    G_pos/G_neg are None when there is no selected/non-selected position.
    """
    p = prob.float()
    pc = p.clamp(1e-7, 1 - 1e-7)
    h_b = float(-(pc * pc.log2() + (1 - pc) * (1 - pc).log2()).mean()) if p.numel() else 0.0
    g_pos = float(p[mask].mean()) if int(mask.sum()) else None
    g_neg = float(p[~mask].mean()) if int((~mask).sum()) else None
    return h_b, g_pos, g_neg


def segment_patches(text_bytes: bytes, boundary_positions: list[int]) -> list[str]:
    """Split a sentence's UTF-8 bytes into patch strings.

    A boundary at byte offset j starts a new patch at j. Bytes before the
    first boundary form a leading segment (they belong to the patch opened by
    BOS). Mid-character boundaries make the split byte-lossy; such patches
    are decoded with backslashreplace so nothing is silently dropped.
    """
    starts = list(boundary_positions)
    if not starts or starts[0] != 0:
        starts = [0] + starts
    starts.append(len(text_bytes))
    return [
        text_bytes[a:b].decode("utf-8", errors="backslashreplace")
        for a, b in zip(starts[:-1], starts[1:])
    ]


def run(args):
    import torch
    from tqdm import tqdm

    from generate import load_from_pretrained
    from hnet.utils.byte_tokenizer import ByteTokenizer
    from visualize_boundaries import compute_boundaries

    dataset_dir = download_flores(Path(args.data_dir))
    sentences = load_flores_split(dataset_dir, args.lang, args.split)
    print(f"Loaded {len(sentences)} sentences for {args.lang}/{args.split}")

    print("Loading model...")
    model = load_from_pretrained(args.model_path, args.config_path)
    device = next(model.parameters()).device
    tokenizer = ByteTokenizer()

    records = {}
    total_bytes = total_bnd = 0
    entropy_wsum = 0.0
    n_truncated = 0
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for sent_id, text in tqdm(sentences, desc=args.lang, unit="sent"):
            enc = tokenizer.encode([text], add_bos=True)[0]
            ids = enc["input_ids"].tolist()
            if args.max_length and len(ids) > args.max_length:
                ids = ids[: args.max_length]
                n_truncated += 1
            if len(ids) <= 1:
                continue
            input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            stages = compute_boundaries(model, input_ids)

            if not 0 <= args.stage < len(stages):
                sys.exit(f"--stage {args.stage} out of range; model has {len(stages)} stage(s)")

            # Detail stage: drop BOS (forced boundary at position 0) so counts
            # and positions are over text bytes only. For stage 0, mask index
            # i maps to text byte i-1; deeper stages index patch positions of
            # the previous stage instead.
            st = stages[args.stage]
            mask = st["boundary_mask"][1:] if args.stage == 0 else st["boundary_mask"]
            prob = st["boundary_prob"][1:] if args.stage == 0 else st["boundary_prob"]
            positions = mask.nonzero(as_tuple=True)[0].tolist()
            n_pos, n_bnd = int(mask.numel()), int(mask.sum())

            text_bytes = bytes(ids[1:])  # sentence bytes as seen by the model
            h_b, g_pos, g_neg = boundary_quality(prob, mask)
            record = {
                "id": sent_id,
                "text": text_bytes.decode("utf-8", errors="backslashreplace"),
                "n_bytes": n_pos if args.stage == 0 else len(text_bytes),
                "n_boundaries": n_bnd,
                "boundary_rate": n_bnd / n_pos if n_pos else 0.0,
                "bytes_per_patch": n_pos / n_bnd if n_bnd else float("inf"),
                "mean_boundary_prob": float(prob.mean()) if n_pos else 0.0,
                "boundary_entropy_bits": h_b,
                "mean_prob_selected": g_pos,
                "mean_prob_unselected": g_neg,
                "boundary_positions": positions,
                # Compact per-stage counts (BOS included, as the model sees it)
                "stage_counts": [
                    [int(s["boundary_mask"].numel()), int(s["boundary_mask"].sum())]
                    for s in stages
                ],
            }
            if args.stage == 0:
                record["patches"] = segment_patches(text_bytes, positions)
            if args.save_probs:
                record["boundary_probs"] = [round(p, 4) for p in prob.tolist()]
            records[sent_id] = record
            total_bytes += n_pos
            total_bnd += n_bnd
            entropy_wsum += h_b * n_pos

    if n_truncated:
        print(f"Warning: {n_truncated} sentences truncated to {args.max_length} bytes")

    rate = total_bnd / total_bytes if total_bytes else float("nan")
    per_sent_bpp = [r["bytes_per_patch"] for r in records.values() if r["n_boundaries"]]
    results = {
        "meta": {
            "dataset": "FLORES-200",
            "split": args.split,
            "lang": args.lang,
            "flores_code": FLORES_ALIASES.get(args.lang, args.lang),
            "model_path": str(args.model_path),
            "config_path": str(args.config_path),
            "stage": args.stage,
            "n_sentences": len(records),
            "total_bytes": total_bytes,
            "total_boundaries": total_bnd,
            "boundary_rate": rate,
            "bytes_per_patch": (1.0 / rate) if rate else float("inf"),
            "mean_sentence_bytes_per_patch": (
                sum(per_sent_bpp) / len(per_sent_bpp) if per_sent_bpp else float("nan")
            ),
            # Byte-weighted mean binary entropy of boundary probs (bits)
            "boundary_entropy_bits": (
                entropy_wsum / total_bytes if total_bytes else float("nan")
            ),
        },
        "sentences": records,
    }

    out = Path(args.output) if args.output else Path(
        f"flores_boundaries_{Path(args.model_path).parent.name}_"
        f"{Path(args.model_path).stem}_{args.lang}_{args.split}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")

    m = results["meta"]
    print(f"\n{args.lang}/{args.split} (stage {args.stage}): "
          f"{m['n_sentences']} sentences, {m['total_bytes']:,} bytes, "
          f"{m['total_boundaries']:,} boundaries")
    print(f"  boundary rate: {m['boundary_rate']:.4f}   "
          f"compression: {m['bytes_per_patch']:.2f} bytes/patch   "
          f"(per-sentence mean {m['mean_sentence_bytes_per_patch']:.2f})")
    print(f"Wrote {out}")


def _label(results):
    m = results["meta"]
    return f"{Path(m['model_path']).parent.name} @ {m['lang']}/{m['split']}"


def compare(args):
    loaded = []
    for path in args.files:
        results = json.loads(Path(path).read_text(encoding="utf-8"))
        loaded.append((path, results))

    print(f"{'file':<50} {'lang':<12} {'#sent':>6} {'bnd/byte':>9} {'bytes/patch':>12}")
    print("-" * 95)
    for path, results in loaded:
        m = results["meta"]
        print(f"{Path(path).name:<50} {m['lang']:<12} {m['n_sentences']:>6} "
              f"{m['boundary_rate']:>9.4f} {m['bytes_per_patch']:>11.2f}x")

    for sent_id in args.id or []:
        print(f"\n=== sentence {sent_id} ===")
        for path, results in loaded:
            rec = results["sentences"].get(str(sent_id)) or results["sentences"].get(sent_id)
            if rec is None:
                print(f"[{_label(results)}] sentence {sent_id} not in {path}")
                continue
            print(f"\n[{_label(results)}]")
            extras = ""
            if rec.get("boundary_entropy_bits") is not None:
                extras += f"  H_b={rec['boundary_entropy_bits']:.3f}b"
            if rec.get("mean_prob_selected") is not None and rec.get("mean_prob_unselected") is not None:
                extras += (f"  G_pos/G_neg={rec['mean_prob_selected']:.3f}"
                           f"/{rec['mean_prob_unselected']:.3f}")
            print(f"  bytes={rec['n_bytes']}  boundaries={rec['n_boundaries']}  "
                  f"bytes/patch={rec['bytes_per_patch']:.2f}  "
                  f"mean_p={rec['mean_boundary_prob']:.3f}{extras}")
            if "patches" in rec:
                print("  " + "|".join(rec["patches"]))
            else:
                print(f"  boundary positions: {rec['boundary_positions']}")


def main():
    parser = argparse.ArgumentParser(
        description="Per-sentence patch boundaries on FLORES-200 (see module docstring)."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Compute boundaries for one checkpoint + language")
    p_run.add_argument("--model-path", required=True,
                       help="Path to the model checkpoint (.pt file)")
    p_run.add_argument("--config-path", required=True,
                       help="Path to the model configuration (.json file)")
    p_run.add_argument("--lang", required=True,
                       help="Language code, e.g. eng_Latn, bul_Cyrl, cmn_Hans "
                            "(HPLT-style aliases accepted)")
    p_run.add_argument("--split", default="devtest", choices=["dev", "devtest"])
    p_run.add_argument("--stage", type=int, default=0,
                       help="Routing stage for per-sentence detail (0 = byte-level)")
    p_run.add_argument("--data-dir", default="/projects/0/hpmlprjs/interns/marko/hnet/data",
                       help="Directory holding/receiving the FLORES-200 download")
    p_run.add_argument("--max-length", type=int, default=4096,
                       help="Truncate sentences longer than this many bytes")
    p_run.add_argument("--save-probs", action="store_true",
                       help="Also store per-position boundary probabilities")
    p_run.add_argument("--output", default=None,
                       help="Output JSON (default: derived from model path + lang + split)")
    p_run.set_defaults(func=run)

    p_cmp = sub.add_parser("compare", help="Compare result files per sentence id")
    p_cmp.add_argument("files", nargs="+", help="Result JSONs from `run`")
    p_cmp.add_argument("--id", type=int, nargs="*",
                       help="Sentence id(s) to show side by side")
    p_cmp.set_defaults(func=compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
