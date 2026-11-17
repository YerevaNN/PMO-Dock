"""Pluggable generation backends for the genetic search loop.

Two backends are provided, both returning the **same thing**: for each input
prompt, the full text ``prompt + completion`` (so downstream
``find_valid_mols`` sees the ``[SMILES]`` start token that lives at the end of
the prompt).

``hf``
    The historical path: ``transformers`` ``model.generate`` in-process, one
    model copy per run process. Kept as the reproducibility baseline and still
    the default, so every existing config behaves exactly as before.

``vllm``
    Talks HTTP to a vLLM OpenAI-compatible server (``/v1/completions``). The
    server is started once per node and shared by every seed process, so one
    set of weights and one continuous-batching scheduler serve all seeds
    instead of N processes time-slicing a GPU.

Ported from the ``Even-More-PMO`` ``genetic_chemalactica`` implementation,
where the design was validated against HF generation on 400 identical prompts
(validity 0.9900 -> 0.9725, uniqueness 0.8232 -> 0.8355, QED and SA
indistinguishable at KS p=0.85) and measured at 3.0-5.1x end-to-end wall time
including server startup. See ``reports/`` for this repo's own verification.

Parity notes:

* Prompts are tokenized **locally** with the run's own tokenizer and sent as
  token ids, so the server cannot silently re-tokenize the prompt. The local
  call mirrors ``HFBackend`` exactly, including ``add_special_tokens=True``:
  the HF path prepends BOS, and a Llama model that does not get one emits EOS
  immediately, yielding empty generations.
* ``stop_at_mol_end`` makes vLLM stop at the first ``[/SMILES]`` and keep it in
  the output. ``find_valid_mols`` only ever reads up to that first end tag, so
  the extracted molecule is unchanged while the tokens after it are never
  generated.

One deliberate divergence from the source implementation: ``genetic_chemllama``
builds prompts that begin directly with the similarity/property tags and end
with ``mol_tags[0]``, with no leading EOS. (``genetic_chemalactica`` prepends
``tokenizer.eos_token`` for Chemlactica checkpoints.) The startup self-check
below mirrors *this* repo's prompt shape accordingly.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Sequence

logger = logging.getLogger(__name__)

try:
    import requests
except Exception as _e:  # pragma: no cover - requests is present in run envs
    requests = None  # type: ignore
    _requests_import_error = _e


# ----------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------


def _resolve_gen_batch_size(model_args, n_prompts: int, default: int = 32) -> int:
    """Mirror the historical in-loop batching rule exactly.

    The original loop read ``args.model.gen_batch_size`` and clamped it to the
    number of prompts, treating ``None`` as "one batch".
    """
    gen_batch_size = getattr(model_args, "gen_batch_size", default)
    if gen_batch_size is None or gen_batch_size > n_prompts:
        gen_batch_size = n_prompts
    return max(1, int(gen_batch_size))


class GenerationBackend:
    """Interface: ``generate(prompts, gen_config) -> list[str]`` (full texts)."""

    name = "base"

    def generate(self, prompts: Sequence[str], gen_config: Dict[str, Any]) -> List[str]:
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - optional hook
        pass


# ----------------------------------------------------------------------------
# HuggingFace (baseline)
# ----------------------------------------------------------------------------


class HFBackend(GenerationBackend):
    """In-process ``transformers`` generation — unchanged from the original loop.

    This is a verbatim lift of the loop body that used to live inline in
    ``run.py``: same batching, same ``batch_decode`` (``skip_special_tokens``
    left at its ``False`` default), same per-batch ``empty_cache()``.
    """

    name = "hf"

    def __init__(self, model, tokenizer, device, model_args):
        import torch  # local import so the vllm path needs no torch

        self._torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_args = model_args

    def generate(self, prompts: Sequence[str], gen_config: Dict[str, Any]) -> List[str]:
        prompts = list(prompts)
        if not prompts:
            return []
        torch = self._torch
        gen_batch_size = _resolve_gen_batch_size(self.model_args, len(prompts))

        all_texts: List[str] = []
        for start_idx in range(0, len(prompts), gen_batch_size):
            batch_prompts = prompts[start_idx : start_idx + gen_batch_size]
            data = self.tokenizer(batch_prompts, return_tensors="pt", padding=True).to(
                self.device
            )
            generated = self.model.generate(**data, **gen_config)
            all_texts.extend(self.tokenizer.batch_decode(generated))
            del data
            del generated
            torch.cuda.empty_cache()
        return all_texts


# ----------------------------------------------------------------------------
# vLLM HTTP server
# ----------------------------------------------------------------------------


class VLLMServerBackend(GenerationBackend):
    """Generation via a shared vLLM OpenAI-compatible server.

    Returns ``prompt + completion`` so the output is drop-in compatible with
    what ``HFBackend`` (and therefore ``find_valid_mols``) expects.
    """

    name = "vllm"

    def __init__(
        self,
        base_url: str,
        tokenizer,
        model_name: str | None = None,
        mol_start_token: str = "[SMILES]",
        mol_end_token: str = "[/SMILES]",
        stop_at_mol_end: bool = True,
        request_batch_size: int = 0,
        seed: int | None = None,
        add_special_tokens: bool = True,
        connect_timeout: int = 30,
        read_timeout: int = 1800,
        max_retries: int = 5,
        retry_sleep: float = 5.0,
    ):
        if requests is None:
            raise ImportError(
                "requests is required for the vllm backend. "
                f"Original error: {_requests_import_error}"
            )
        self.base_url = self._normalize_base_url(base_url)
        self.completions_url = f"{self.base_url}/v1/completions"
        self.models_url = f"{self.base_url}/v1/models"
        self.tokenizer = tokenizer
        self.mol_start_token = mol_start_token
        self.mol_end_token = mol_end_token
        self.stop_at_mol_end = bool(stop_at_mol_end)
        self.request_batch_size = int(request_batch_size or 0)
        self.seed = seed
        self.add_special_tokens = bool(add_special_tokens)
        self._timeout = (connect_timeout, read_timeout)
        self.max_retries = int(max_retries)
        self.retry_sleep = float(retry_sleep)
        self._call_count = 0

        self.model_name = model_name or self._discover_model_name()
        logger.info(
            "VLLMServerBackend: url=%s model=%s stop_at_mol_end=%s",
            self.completions_url,
            self.model_name,
            self.stop_at_mol_end,
        )

    # -- server handshake ---------------------------------------------------

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        base_url = str(base_url).strip().rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            base_url = f"http://{base_url}"
        if base_url.endswith("/v1"):
            base_url = base_url[: -len("/v1")]
        return base_url

    def _discover_model_name(self) -> str:
        res = requests.get(self.models_url, timeout=self._timeout)
        res.raise_for_status()
        data = res.json().get("data") or []
        if not data:
            raise RuntimeError(f"vLLM server at {self.models_url} reports no models")
        return data[0]["id"]

    @classmethod
    def wait_until_ready(
        cls, base_url: str, timeout_s: float = 1800.0, poll_s: float = 5.0
    ) -> None:
        """Block until the server answers ``/v1/models`` (or raise)."""
        if requests is None:
            raise ImportError("requests is required to wait on a vLLM server")
        url = f"{cls._normalize_base_url(base_url)}/v1/models"
        deadline = time.time() + timeout_s
        last_err: Exception | None = None
        while time.time() < deadline:
            try:
                res = requests.get(url, timeout=(5, 10))
                if res.status_code == 200:
                    logger.info("vLLM server ready at %s", url)
                    return
                last_err = RuntimeError(f"status {res.status_code}")
            except Exception as e:  # server still booting
                last_err = e
            time.sleep(poll_s)
        raise RuntimeError(f"vLLM server at {url} not ready within {timeout_s}s: {last_err}")

    # -- sampling param translation ----------------------------------------

    def _build_payload(
        self, token_id_batch: List[List[int]], gen_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        cfg = dict(gen_config or {})
        do_sample = cfg.pop("do_sample", True)
        max_tokens = cfg.pop("max_new_tokens", None)
        n = cfg.pop("num_return_sequences", 1)
        temperature = cfg.pop("temperature", 1.0)

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": token_id_batch,
            "n": int(n or 1),
            "temperature": float(temperature) if do_sample else 0.0,
        }
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        for key in ("top_p", "top_k", "repetition_penalty", "presence_penalty",
                    "frequency_penalty", "min_p", "length_penalty"):
            if key in cfg:
                payload[key] = cfg.pop(key)
        if self.seed is not None:
            # NOTE: one seed covers every prompt in the request, which correlates
            # sampling across the batch and measurably lowers uniqueness (0.82 ->
            # 0.74 on the 400-prompt parity check). Prefer seeding the SERVER
            # (--seed) and leaving this None. Kept for exact-replay debugging.
            payload["seed"] = int(self.seed) + self._call_count
        # HFBackend decodes with tokenizer.batch_decode(...), whose
        # skip_special_tokens default is False. vLLM's OpenAI server defaults to
        # True, which mangles the tag delimiters this model relies on
        # ("[SMILES]CCO[/SMILES]" -> "SMILESCCOSMILES"). find_valid_mols then
        # finds no end tag and discards every molecule, so this must stay False.
        payload["skip_special_tokens"] = False
        if self.stop_at_mol_end:
            payload["stop"] = [self.mol_end_token]
            payload["include_stop_str_in_output"] = True
        if cfg:
            logger.debug("vLLM backend ignoring unmapped generation keys: %s", sorted(cfg))
        return payload

    # -- request ------------------------------------------------------------

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                res = requests.post(self.completions_url, json=payload, timeout=self._timeout)
                if res.status_code == 200:
                    return res.json()
                last_err = RuntimeError(f"status {res.status_code}: {res.text[:300]}")
                logger.warning("vLLM request failed (%s), retrying...", last_err)
            except Exception as e:
                last_err = e
                logger.warning("vLLM request error: %s, retrying...", str(e)[:200])
            time.sleep(self.retry_sleep * (attempt + 1))
        raise RuntimeError(f"vLLM request failed after {self.max_retries} attempts: {last_err}")

    def self_check(self) -> None:
        """Generate one molecule and confirm it survives the round trip.

        Two real bugs got through code review on the original implementation: a
        missing BOS (empty generations) and vLLM's ``skip_special_tokens``
        default mangling the tag delimiters (non-empty generations, zero
        extractable molecules). Both looked like enormous speedups. This costs
        about a second at startup and turns either failure into an immediate,
        explicit error.

        Both failures this guards against are DETERMINISTIC: no BOS gives an
        empty completion every time, and skip_special_tokens strips the tag
        every time. Sampling, by contrast, is not deterministic -- a probe can
        legitimately run out of tokens mid-molecule and so carry no end tag
        while the server is configured perfectly. A single probe cannot tell
        those apart, and treating one unlucky draw as fatal killed 2 of 5 seeds
        in an otherwise healthy run while the other 3 completed against the
        SAME server.

        So: draw several probes with real headroom, and fail only if none of
        them closes a molecule. A genuine misconfiguration still fails every
        probe and is still caught on the first run.
        """
        # Mirror this repo's prompt shape: prompts end with mol_tags[0] and
        # carry no leading EOS; BOS is added by the tokenizer.
        probe = self.mol_start_token
        attempts, completions = 4, []
        for _ in range(attempts):
            out = self.generate([probe], {"max_new_tokens": 256, "temperature": 1.0})
            text = out[0] if out else ""
            completion = text[len(probe):]
            completions.append(completion)
            if self.mol_end_token in completion:
                logger.info("vLLM self-check passed: %r", completion[:80])
                return
        # Unreachable in practice: generate() rejects an all-empty batch before
        # we get here, and its message already names BOS as the likely cause.
        # Kept as a backstop in case that guard is ever relaxed.
        if not any(completions):
            raise RuntimeError(
                f"vLLM self-check failed: all {attempts} probes came back empty. "
                "This usually means the prompt reached the server without a BOS "
                "token, which makes a Llama model emit EOS immediately."
            )
        raise RuntimeError(
            f"vLLM self-check failed: none of {attempts} probes contained "
            f"{self.mol_end_token!r}, so find_valid_mols would discard every "
            "molecule. This usually means the server detokenized with "
            "skip_special_tokens=True (which mangles the tag delimiters). "
            f"Last completion was: {completions[-1][:200]!r}"
        )

    def generate(self, prompts: Sequence[str], gen_config: Dict[str, Any]) -> List[str]:
        prompts = list(prompts)
        if not prompts:
            return []

        # Tokenize locally so the server never re-tokenizes, and mirror the HF
        # path EXACTLY: HFBackend calls ``tokenizer(batch_prompts, ...)`` with
        # add_special_tokens left at its default (True), which prepends BOS.
        # Dropping BOS makes a Llama model emit EOS immediately -> empty
        # generations, so this must not be "cleverly" disabled.
        token_ids: List[List[int]] = self.tokenizer(
            prompts, add_special_tokens=self.add_special_tokens
        )["input_ids"]

        step = self.request_batch_size if self.request_batch_size > 0 else len(prompts)
        completions: List[str] = [""] * len(prompts)

        for start_idx in range(0, len(prompts), step):
            end_idx = min(start_idx + step, len(prompts))
            payload = self._build_payload(token_ids[start_idx:end_idx], gen_config)
            data = self._post(payload)
            self._call_count += 1

            n = int(payload.get("n", 1))
            choices = data.get("choices") or []
            if len(choices) != (end_idx - start_idx) * n:
                raise RuntimeError(
                    f"vLLM returned {len(choices)} choices for "
                    f"{(end_idx - start_idx) * n} expected"
                )
            # With n>1 vLLM emits n consecutive choices per prompt; keep the
            # first for each so ordering matches the prompt list.
            for choice in choices:
                idx = int(choice["index"])
                if idx % n != 0:
                    continue
                completions[start_idx + idx // n] = choice.get("text", "")

        # A batch where the model emitted nothing at all is never legitimate: it
        # means the request was malformed (e.g. a missing BOS makes a Llama model
        # emit EOS immediately). Without this check the run looks blazingly fast
        # and silently produces no molecules for hours.
        if not any(c.strip() for c in completions):
            raise RuntimeError(
                f"vLLM returned {len(completions)} empty completions for "
                f"{len(prompts)} prompts. The request is likely malformed — check "
                f"that prompt tokenization matches the HF path (add_special_tokens="
                f"{self.add_special_tokens}, so BOS is "
                f"{'present' if self.add_special_tokens else 'ABSENT'}), and that "
                f"stop={self.mol_end_token!r} is not matching at position 0."
            )

        return [p + c for p, c in zip(prompts, completions, strict=True)]


# ----------------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------------


def build_backend(
    backend_name: str,
    *,
    model=None,
    tokenizer=None,
    device=None,
    model_args=None,
    vllm_url: str | None = None,
    vllm_model_name: str | None = None,
    stop_at_mol_end: bool = True,
    request_batch_size: int = 0,
    seed: int | None = None,
    wait_for_server: bool = True,
    server_wait_s: float = 1800.0,
    self_check: bool = True,
) -> GenerationBackend:
    backend_name = (backend_name or "hf").strip().lower()

    if backend_name == "hf":
        if model is None or tokenizer is None:
            raise ValueError("hf backend requires model and tokenizer")
        return HFBackend(model, tokenizer, device, model_args)

    if backend_name == "vllm":
        url = vllm_url or os.environ.get("VLLM_SERVICE_URL", "").strip()
        if not url:
            raise ValueError(
                "vllm backend requires generation_backend_url (or VLLM_SERVICE_URL)"
            )
        if wait_for_server:
            VLLMServerBackend.wait_until_ready(url, timeout_s=server_wait_s)
        mol_start, mol_end = "[SMILES]", "[/SMILES]"
        if model_args is not None:
            try:
                mol_start, mol_end = model_args.mol_tags[0], model_args.mol_tags[1]
            except Exception:
                pass
        backend = VLLMServerBackend(
            base_url=url,
            tokenizer=tokenizer,
            model_name=vllm_model_name,
            mol_start_token=mol_start,
            mol_end_token=mol_end,
            stop_at_mol_end=stop_at_mol_end,
            request_batch_size=request_batch_size,
            seed=seed,
        )
        if self_check:
            backend.self_check()
        return backend

    raise ValueError(f"Unknown generation backend {backend_name!r} (expected 'hf' or 'vllm')")
