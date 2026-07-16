# H-Net Multilingual Eval Suite

One-page map of every evaluation pipeline in this repo: what it measures, how it
scores, and how to run and aggregate it.

## Pipelines at a glance

| Pipeline | Entry point | Languages | Scoring | Model access |
|---|---|---|---|---|
| **EuroEval** | `eval/run_euroeval.py` | en, bg, nl, fi (no zh) | cloze (default) or generative per task | HTTP server (`eval/hnet_openai_server.py`), two envs |
| **CUTE / EXECUTE** | `eval/run_cute.py` | en | generative EM / char-similarity / char-F1 | direct (hnet env) |
| **FLORES MT** | `multilingual_translation/` + `jobs/.../flores` | pairs | translation metrics | direct |
| **lm-eval-harness (EU20/OLMES)** | `eval/run_benchmarks.py` via `eval/lm_eval_wrapper.py` | en | standard harness log-prob | direct |

The EuroEval pipeline is the primary multilingual suite; the rest are
complementary probes (character-level skills, MT, English-standard baselines).

## EuroEval pipeline architecture

EuroEval cannot import in an env containing `flash_attn`, and H-Net requires it,
so the pipeline is split (see `requirements-euroeval.txt`):

- **Server** (`hnet` env): `eval/hnet_openai_server.py` — OpenAI-compatible
  FastAPI server. Endpoints: `/v1/completions`, `/v1/chat/completions`,
  `/v1/models`, and the non-standard **`/v1/loglikelihood`** which scores
  candidate continuations (log-prob restricted to exactly the continuation
  bytes, conditioned on the prompt; returns raw sum, `n_bytes`, and per-byte
  log-probs).
- **Driver** (`euroeval` env): `eval/run_euroeval.py` — imports upstream
  EuroEval (17.2.0) and adapts it to the byte-level server via runtime patches.

### Scoring modes

**Cloze scoring (`--cloze-scoring`, default ON)** — applies to
sequence-classification (sentiment, linguistic-acceptability) and
multiple-choice (knowledge, common-sense) task groups:

1. For MC tasks, the **full choice texts** are parsed from the prompt
   (`parse_mc_choices`; validated against 1886 real prompts across en/bg
   dumps). For classification tasks, candidates are the localized label words
   (e.g. `да`/`не` for scala-bg).
2. Each candidate is scored via `/v1/loglikelihood` — loss on the answer bytes
   only.
3. Selection by **byte-normalized log-prob** (`--cloze-norm byte`, default) —
   the per-byte mean removes the length penalty of raw sums. `--cloze-norm
   none` reverts to raw summed log-prob.
4. All candidate scores (raw, n_bytes, per-byte) are dumped to
   `<output-dir>/cloze_scores/<model>-<dataset>.jsonl` for post-hoc analysis
   (e.g. gold-answer BPB as a smooth small-model diagnostic, alternative
   normalizations).

Note: upstream EuroEval ≥17.6 has a `--use-bits-per-character` "cloze" mode,
but it is a gold-answer perplexity metric (no per-candidate argmax), hard-coded
to **character** normalization, and only works on the vLLM CUDA backend — it
cannot drive an HTTP-served byte model. Hence this local implementation.

**Generative tasks** (NER, reading-comprehension, summarization) go through the
normal completions path. The server multiplies the client's `max_tokens` budget
by **`--gen-byte-scale` (default 4.0)** because EuroEval budgets in subword
tokens while H-Net generates bytes — without this, structured outputs (NER
JSON) get truncated mid-answer and score 0 as "failed instances". Hard cap:
`--max-gen-tokens` (default 4096 bytes).

### Model → language matrix (S-size comparison suite)

Defined in `/gpfs/home3/mivanov1/jobs/hnet/eval/euroeval/eval_hnet_euroeval.job`
(NAMES/LANGS arrays, one SLURM array task per model):

| Model | EuroEval languages | Note |
|---|---|---|
| `hnet_1stage_S_en` | en | baseline |
| `hnet_1stage_S_en_bul` | en, bg | |
| `hnet_1stage_S_en_cmn` | en | **EuroEval has no Chinese benchmarks** — use CUTE/FLORES/bpb for the zh side |
| `hnet_1stage_S_en_fin` | en, fi | |
| `hnet_1stage_S_en_nld` | en, nl | |

Datasets per language are chosen by EuroEval itself (all official benchmarks
for the requested languages) unless `--datasets`/`--tasks` restricts them.

### Running

```bash
# Full array over the S models (server + driver per task):
sbatch /gpfs/home3/mivanov1/jobs/hnet/eval/euroeval/eval_hnet_euroeval.job

# Manual single run (two terminals / two envs):
# hnet env:
python eval/hnet_openai_server.py --model-path .../latest.pt \
    --config-path configs/comparison/S/hnet_1stage_S.json --port 8765
# euroeval env:
python eval/run_euroeval.py --api-base http://localhost:8765/v1 \
    --model-name hnet_1stage_S_en_bul --languages en,bg \
    --output-dir eval/results/euroeval/S --debug --force
```

**`--limit N` truncates every split to N samples — debug only.** The existing
S results were produced with `--limit 20`; standard errors of ±3–10 MCC points
make model comparisons meaningless at that size. Drop `--limit` (or use ≥ the
full validation split) for reportable numbers.

### Outputs & aggregation

| File | Producer | Content |
|---|---|---|
| `eval/results/euroeval/**/euroeval_<model>_<ts>.jsonl` | driver | one line per (dataset, task): metrics + SEs |
| `.../<model>-<dataset>-base-model-outputs.json` | EuroEval `--debug` | per-sample prompts, generations, gold labels |
| `.../cloze_scores/<model>-<dataset>.jsonl` | cloze patch | per-sample candidate scores (raw / n_bytes / per-byte) |
| `euroeval_benchmark_results.jsonl` (CWD) | EuroEval | canonical append-only log |

Aggregate into one model × dataset table (primary metric ± SE per task,
`!` = failed instances):

```bash
python eval/aggregate_euroeval.py --results-dir eval/results/euroeval/S [--csv out.csv]
```

### Reading the scores

- **MCC** is the primary metric for all classification/MC tasks: 0 = chance,
  regardless of label imbalance. Ignore accuracy/macro-F1 when MCC ≈ 0 — e.g.
  61% accuracy on a skewed binary task with MCC ≈ 0 is a majority-class
  artifact, not capability.
- At small scale (≤S) expect near-chance MCC on few-shot tasks; the smooth
  discriminative signals are bpb / per-language val loss (wandb), the
  cloze-score sidecars (gold-answer per-byte log-prob), and chrF on
  summarization.
