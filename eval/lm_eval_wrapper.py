"""
lm-evaluation-harness wrapper for HNet byte-level language models.

Implements the TemplateLM interface so HNet can be evaluated on standard
benchmarks (LAMBADA, HellaSwag, PIQA, ARC, WinoGrande, OpenBookQA).

Since HNet operates on raw UTF-8 bytes (vocab_size=256, BOS=254, EOS=255),
the wrapper handles string <-> byte-token conversion transparently.
"""

import torch
import torch.nn.functional as F
from typing import Optional
from tqdm import tqdm

from lm_eval.api.model import TemplateLM
from lm_eval.api.instance import Instance

from generate import load_from_pretrained
from hnet.utils.tokenizers import ByteTokenizer


class HNetLM(TemplateLM):
    """lm-evaluation-harness wrapper for HNet."""

    def __init__(
        self,
        model_path: str,
        config_path: str,
        batch_size: int = 1,
        max_length: int = 8192,
        device: str = "cuda",
    ):
        super().__init__()
        self.model = load_from_pretrained(model_path, config_path)
        self.model.eval()
        self._device = torch.device(device)
        self.tokenizer = ByteTokenizer()
        self._batch_size = batch_size
        self._max_length = max_length
        self.vocab_size = self.tokenizer.vocab_size

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_idx

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return 1024

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str, add_special_tokens: bool = False, **kwargs) -> list[int]:
        """Encode string to byte token IDs."""
        encoded = self.tokenizer.encode(
            [string], add_bos=add_special_tokens, add_eos=False
        )[0]
        return encoded["input_ids"].tolist()

    def tok_decode(self, tokens: list[int], **kwargs) -> str:
        """Decode byte token IDs to string."""
        # Filter out BOS/EOS tokens
        tokens = [t for t in tokens if t not in (self.tokenizer.bos_idx, self.tokenizer.eos_idx)]
        try:
            return self.tokenizer.decode(tokens)
        except UnicodeDecodeError:
            return self.tokenizer.decode(tokens, errors="replace")

    def _model_call(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run forward pass and return logits. input_ids: (B, L)."""
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = self.model(input_ids, mask=None)
        return output.logits.float()

    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str], list[int], list[int]]],
        disable_tqdm: bool = False,
        override_bs: Optional[int] = None,
    ) -> list[tuple[float, bool]]:
        """Score pre-tokenized (context, continuation) pairs.

        Each request is: ((context_str, cont_str), context_ids, cont_ids)
        Returns: list of (log_prob, is_greedy) tuples
        """
        results = []
        batch_size = override_bs or self._batch_size

        # Sort by total length descending for efficient batching
        sorted_requests = sorted(
            enumerate(requests),
            key=lambda x: -(len(x[1][1]) + len(x[1][2])),
        )

        for i in tqdm(
            range(0, len(sorted_requests), batch_size),
            desc="loglikelihood",
            disable=disable_tqdm,
        ):
            batch_indices = []
            batch_input_ids = []
            batch_cont_lens = []

            for orig_idx, (_, ctx_ids, cont_ids) in sorted_requests[i : i + batch_size]:
                full_ids = ctx_ids + cont_ids
                # Truncate from the left if too long
                if len(full_ids) > self._max_length:
                    full_ids = full_ids[-self._max_length:]
                    # Adjust cont_len if context was fully truncated
                    cont_len = min(len(cont_ids), len(full_ids))
                else:
                    cont_len = len(cont_ids)

                batch_indices.append(orig_idx)
                batch_input_ids.append(full_ids)
                batch_cont_lens.append(cont_len)

            # Pad to same length within batch
            max_len = max(len(ids) for ids in batch_input_ids)
            padded = torch.full(
                (len(batch_input_ids), max_len),
                0,  # pad with 0 (null byte)
                dtype=torch.long,
                device=self._device,
            )
            for j, ids in enumerate(batch_input_ids):
                padded[j, : len(ids)] = torch.tensor(ids, dtype=torch.long)

            logits = self._model_call(padded)  # (B, L, V)

            for j in range(len(batch_input_ids)):
                cont_len = batch_cont_lens[j]
                seq_len = len(batch_input_ids[j])

                # Get logits for positions that predict continuation tokens
                # Model predicts token[t+1] at position t
                # So for continuation tokens at positions [seq_len-cont_len : seq_len],
                # we need logits at positions [seq_len-cont_len-1 : seq_len-1]
                cont_logits = logits[j, seq_len - cont_len - 1 : seq_len - 1, :]  # (cont_len, V)
                cont_tokens = padded[j, seq_len - cont_len : seq_len]  # (cont_len,)

                log_probs = F.log_softmax(cont_logits, dim=-1)
                token_log_probs = log_probs.gather(
                    dim=-1, index=cont_tokens.unsqueeze(-1)
                ).squeeze(-1)

                total_log_prob = token_log_probs.sum().item()

                # Check if greedy decode matches continuation
                greedy_tokens = cont_logits.argmax(dim=-1)
                is_greedy = (greedy_tokens == cont_tokens).all().item()

                results.append((batch_indices[j], (total_log_prob, is_greedy)))

        # Restore original order
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def loglikelihood_rolling(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[tuple[float, bool]]:
        """Compute rolling log-likelihood over full documents (for perplexity)."""
        results = []

        for req in tqdm(requests, desc="loglikelihood_rolling", disable=disable_tqdm):
            (string,) = req.args
            token_ids = self.tok_encode(string)

            total_log_prob = 0.0
            is_greedy = True

            # Process in chunks if longer than max_length
            for start in range(0, len(token_ids), self._max_length):
                chunk = token_ids[start : start + self._max_length]
                input_ids = torch.tensor(
                    [chunk], dtype=torch.long, device=self._device
                )
                logits = self._model_call(input_ids)  # (1, L, V)

                # Predict each token from the previous one
                shift_logits = logits[0, :-1, :]  # (L-1, V)
                shift_targets = input_ids[0, 1:]  # (L-1,)

                log_probs = F.log_softmax(shift_logits, dim=-1)
                token_log_probs = log_probs.gather(
                    dim=-1, index=shift_targets.unsqueeze(-1)
                ).squeeze(-1)

                total_log_prob += token_log_probs.sum().item()

                greedy_tokens = shift_logits.argmax(dim=-1)
                if not (greedy_tokens == shift_targets).all().item():
                    is_greedy = False

            results.append((total_log_prob, is_greedy))

        return results

    def generate_until(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[str]:
        """Generate text until stop conditions are met."""
        results = []

        for req in tqdm(requests, desc="generate_until", disable=disable_tqdm):
            context, gen_kwargs = req.args
            until = gen_kwargs.get("until", [])
            max_gen_toks = gen_kwargs.get("max_gen_toks", self.max_gen_toks)

            context_ids = self.tok_encode(context)
            if len(context_ids) > self._max_length:
                context_ids = context_ids[-self._max_length:]

            input_ids = torch.tensor(
                [context_ids], dtype=torch.long, device=self._device
            )

            # Prefill
            inference_cache = self.model.allocate_inference_cache(
                1, len(context_ids) + max_gen_toks, dtype=torch.bfloat16
            )
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                mask = torch.ones(input_ids.shape, device=self._device, dtype=torch.bool)
                output = self.model.forward(
                    input_ids, mask=mask, inference_params=inference_cache
                )

            generated_ids = []
            logits = output.logits[0, -1, :]

            for _ in range(max_gen_toks):
                next_token = logits.argmax(dim=-1).item()

                if next_token == self.tokenizer.eos_idx:
                    break

                generated_ids.append(next_token)

                # Check stop conditions
                try:
                    generated_text = self.tokenizer.decode(generated_ids, errors="replace")
                    if any(stop in generated_text for stop in until):
                        break
                except (UnicodeDecodeError, ValueError):
                    pass

                current_token = torch.tensor(
                    [[next_token]], dtype=torch.long, device=self._device
                )
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = self.model.step(current_token, inference_cache)
                logits = output.logits[0, -1, :]

            try:
                generated_text = self.tokenizer.decode(generated_ids, errors="replace")
            except (UnicodeDecodeError, ValueError):
                generated_text = ""

            # Trim at first stop string
            for stop in until:
                if stop in generated_text:
                    generated_text = generated_text[: generated_text.index(stop)]
                    break

            results.append(generated_text)

        return results
