"""Differential test: HFBackend must reproduce the old inline generation loop.

The vLLM port moved the generation loop out of ``run.py`` and into
``HFBackend``. That refactor touches **every existing hf run**, which is the
real risk in the change -- a vLLM bug only affects runs that opt in, but a
mistake here silently changes results for the default path that produced every
number in this repo so far.

So this compares the two implementations directly: the loop body exactly as it
stood in ``run.py`` before the port (``_legacy_generate`` below, lifted
verbatim from git history) against ``HFBackend.generate``, on the same model,
the same prompts and the same RNG state. The outputs must be identical
strings, not merely similar distributions.

The property under test -- two implementations, same RNG state, same output --
does not depend on the device, so this runs on CPU (``research_cpu``, minutes,
no queue) and on GPU (confirmation on the device production actually uses).
``DIFF_TEST_DEVICE`` picks; the default is CUDA when present, else CPU. dtype
follows the device: bfloat16 on GPU as in ``run.py``, float32 on CPU where
bfloat16 matmuls are slow.

Launchers: ``genetic_chemllama/slurm/run_hf_differential.slurm`` (GPU) and
``run_hf_differential_cpu.slurm``.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from genetic.generation_backends import HFBackend

MODEL_ID = os.environ.get("DIFF_TEST_MODEL", "yerevann/ChemLlama-170M")
PAD_TOKEN = "<|finetune_right_pad_id|>"
DEVICE = os.environ.get("DIFF_TEST_DEVICE") or (
    "cuda:0" if torch.cuda.is_available() else "cpu"
)
# run.py loads in bfloat16; keep that on GPU. On CPU bfloat16 matmuls are slow
# enough to dominate the job, and the equality being tested is dtype-agnostic.
DTYPE = torch.bfloat16 if DEVICE.startswith("cuda") else torch.float32


def _legacy_generate(model, tokenizer, device, gen_batch_size, prompts, gen_config):
    """The loop body as it stood in run.py before the backend split.

    Lifted verbatim (only the surrounding `with timer.phase(...)` removed) so
    that any behavioural drift in HFBackend shows up as a diff here.
    """
    all_generated_tokens = []
    for start_idx in range(0, len(prompts), gen_batch_size):
        end_idx = min(start_idx + gen_batch_size, len(prompts))
        batch_prompts = prompts[start_idx:end_idx]

        data = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)

        generated_texts = model.generate(**data, **gen_config)
        batch_generated_tokens = tokenizer.batch_decode(generated_texts)
        all_generated_tokens.extend(batch_generated_tokens)

        del data
        del generated_texts
        torch.cuda.empty_cache()
    return all_generated_tokens


class _ModelArgs:
    def __init__(self, gen_batch_size):
        self.gen_batch_size = gen_batch_size


@pytest.fixture(scope="module")
def model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
        device_map={"": DEVICE},
        attn_implementation="sdpa",
    )
    model.eval()
    tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_ID, padding_side="left")
    pad_token_id = tokenizer.convert_tokens_to_ids(PAD_TOKEN)
    if pad_token_id != tokenizer.unk_token_id:
        tokenizer.pad_token_id = pad_token_id
    return model, tokenizer


def _prompts(n):
    """Prompt shape mirrors create_prompts(): tags, then the start token."""
    return [
        f"[SIMILAR]CCO 0.7[/SIMILAR][QED]0.80[/QED][SMILES]" if i % 2 else "[SMILES]"
        for i in range(n)
    ]


GEN_CONFIG = {
    "max_new_tokens": 48,
    "temperature": 1.0,
    "do_sample": True,
    "repetition_penalty": 1.01,
    "num_return_sequences": 1,
}


@pytest.mark.parametrize("gen_batch_size,n_prompts", [(4, 10), (32, 10), (1, 3)])
def test_hf_backend_matches_legacy_loop(model_and_tokenizer, gen_batch_size, n_prompts):
    """Same seed, same prompts -> identical generated text, batch size and all."""
    model, tokenizer = model_and_tokenizer
    prompts = _prompts(n_prompts)

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    legacy = _legacy_generate(
        model, tokenizer, DEVICE, min(gen_batch_size, n_prompts), prompts, dict(GEN_CONFIG)
    )

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    backend = HFBackend(model, tokenizer, DEVICE, _ModelArgs(gen_batch_size))
    ported = backend.generate(prompts, dict(GEN_CONFIG))

    assert len(ported) == len(legacy) == n_prompts
    assert ported == legacy


def test_hf_backend_returns_prompt_plus_completion(model_and_tokenizer):
    """The contract the vLLM backend has to match: full text, prompt included.

    batch_decode keeps special tokens, so the decoded text starts with BOS and
    then the prompt. What matters downstream is that the prompt's trailing
    [SMILES] survives, because find_valid_mols keys on it.
    """
    model, tokenizer = model_and_tokenizer
    prompts = _prompts(4)
    backend = HFBackend(model, tokenizer, DEVICE, _ModelArgs(4))
    out = backend.generate(prompts, dict(GEN_CONFIG))
    assert len(out) == 4
    for text in out:
        assert "[SMILES]" in text
