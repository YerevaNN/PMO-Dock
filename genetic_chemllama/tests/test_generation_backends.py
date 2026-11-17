"""Unit tests for the genetic_chemllama generation backends.

These are deliberately dependency-light: ``generation_backends`` imports only
``requests`` at module scope (``torch`` is imported inside ``HFBackend``), so
this file runs on a CPU node in seconds with no model, no GPU and no server.

What is actually being guarded here is the set of failures the vLLM port hit
during its original development, every one of which was **silent** -- the run
got dramatically faster and produced nothing, or produced molecules that were
quietly discarded downstream:

1. Missing BOS -> a Llama model emits EOS immediately -> empty completions.
2. ``skip_special_tokens=True`` on the server -> "[SMILES]CCO[/SMILES]"
   detokenizes to "SMILESCCOSMILES" -> ``find_valid_mols`` finds no end tag
   and discards every molecule.
3. A single sampled self-check probe treated as authoritative -> healthy seeds
   killed at startup because one draw ran out of tokens mid-molecule.

Plus the structural contract the whole design rests on: both backends return
``prompt + completion``, in prompt order.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genetic import generation_backends as gb
from genetic.generation_backends import (
    VLLMServerBackend,
    _resolve_gen_batch_size,
    build_backend,
)

MOL_START = "[SMILES]"
MOL_END = "[/SMILES]"


class FakeTokenizer:
    """Minimal stand-in: records how it was called, returns one id per char."""

    eos_token = "</s>"

    def __init__(self):
        self.calls = []

    def __call__(self, prompts, add_special_tokens=True, **kwargs):
        self.calls.append({"prompts": list(prompts), "add_special_tokens": add_special_tokens})
        bos = [1] if add_special_tokens else []
        return {"input_ids": [bos + [ord(c) % 100 for c in p] for p in prompts]}


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code != 200:
            raise RuntimeError(f"status {self.status_code}")


def make_backend(monkeypatch, completions_for, **kwargs):
    """Build a VLLMServerBackend whose server returns scripted completions.

    ``completions_for`` is called with the posted payload and returns the list
    of completion strings, one per prompt per ``n``.
    """
    posted = []

    def fake_post(url, json=None, timeout=None):
        posted.append(json)
        texts = completions_for(json)
        return FakeResponse(
            {"choices": [{"index": i, "text": t} for i, t in enumerate(texts)]}
        )

    monkeypatch.setattr(gb.requests, "post", fake_post)
    tokenizer = FakeTokenizer()
    backend = VLLMServerBackend(
        base_url="http://fake:8000",
        tokenizer=tokenizer,
        model_name="fake-model",
        mol_start_token=MOL_START,
        mol_end_token=MOL_END,
        **kwargs,
    )
    return backend, tokenizer, posted


# ---------------------------------------------------------------------------
# Payload construction
# ---------------------------------------------------------------------------


def test_skip_special_tokens_is_always_false(monkeypatch):
    """Regression: the server default (True) mangles the tag delimiters.

    With skip_special_tokens=True the server returns "SMILESCCOSMILES" and
    find_valid_mols discards every molecule while the run looks fast and
    healthy. This flag is not negotiable and no caller may override it.
    """
    backend, _, _ = make_backend(monkeypatch, lambda p: ["x"])
    payload = backend._build_payload([[1, 2]], {"max_new_tokens": 64})
    assert payload["skip_special_tokens"] is False


def test_stop_at_mol_end_keeps_the_end_tag(monkeypatch):
    """The tag must survive into the output or find_valid_mols sees no molecule."""
    backend, _, _ = make_backend(monkeypatch, lambda p: ["x"], stop_at_mol_end=True)
    payload = backend._build_payload([[1]], {})
    assert payload["stop"] == [MOL_END]
    assert payload["include_stop_str_in_output"] is True


def test_stop_can_be_disabled(monkeypatch):
    backend, _, _ = make_backend(monkeypatch, lambda p: ["x"], stop_at_mol_end=False)
    payload = backend._build_payload([[1]], {})
    assert "stop" not in payload
    assert "include_stop_str_in_output" not in payload


def test_hf_generation_keys_are_translated(monkeypatch):
    backend, _, _ = make_backend(monkeypatch, lambda p: ["x"])
    payload = backend._build_payload(
        [[1]],
        {"max_new_tokens": 128, "temperature": 0.9, "top_p": 0.95, "top_k": 50,
         "repetition_penalty": 1.1, "num_return_sequences": 2},
    )
    assert payload["max_tokens"] == 128
    assert payload["temperature"] == pytest.approx(0.9)
    assert payload["top_p"] == pytest.approx(0.95)
    assert payload["top_k"] == 50
    assert payload["repetition_penalty"] == pytest.approx(1.1)
    assert payload["n"] == 2


def test_do_sample_false_means_greedy(monkeypatch):
    backend, _, _ = make_backend(monkeypatch, lambda p: ["x"])
    payload = backend._build_payload([[1]], {"do_sample": False, "temperature": 0.9})
    assert payload["temperature"] == 0.0


def test_no_request_seed_by_default(monkeypatch):
    """A per-request seed correlates sampling across the batch; it must be opt-in."""
    backend, _, _ = make_backend(monkeypatch, lambda p: ["x"])
    assert "seed" not in backend._build_payload([[1]], {})


# ---------------------------------------------------------------------------
# The prompt + completion contract
# ---------------------------------------------------------------------------


def test_generate_returns_prompt_plus_completion_in_order(monkeypatch):
    """HFBackend returns full texts; the vLLM path must be drop-in identical."""
    prompts = [f"{MOL_START}", f"AA{MOL_START}", f"BBB{MOL_START}"]

    def completions(payload):
        return [f"C{i}{MOL_END}" for i in range(len(payload["prompt"]))]

    backend, _, _ = make_backend(monkeypatch, completions)
    out = backend.generate(prompts, {"max_new_tokens": 32})
    assert out == [f"{p}C{i}{MOL_END}" for i, p in enumerate(prompts)]


def test_bos_is_requested_from_the_tokenizer(monkeypatch):
    """Regression: without BOS a Llama model emits EOS immediately."""
    backend, tokenizer, _ = make_backend(monkeypatch, lambda p: [f"C{MOL_END}"])
    backend.generate([MOL_START], {})
    assert tokenizer.calls[-1]["add_special_tokens"] is True


def test_prompts_are_sent_as_token_ids_not_text(monkeypatch):
    """The server must not get a second chance to tokenize."""
    backend, _, posted = make_backend(monkeypatch, lambda p: [f"C{MOL_END}"])
    backend.generate([MOL_START], {})
    sent = posted[-1]["prompt"]
    assert isinstance(sent, list) and isinstance(sent[0], list)
    assert all(isinstance(t, int) for t in sent[0])


def test_all_empty_completions_raise(monkeypatch):
    """Regression: a malformed request yields empty output and looks 50x faster."""
    backend, _, _ = make_backend(monkeypatch, lambda p: ["" for _ in p["prompt"]])
    with pytest.raises(RuntimeError, match="empty completions"):
        backend.generate([MOL_START, MOL_START], {})


def test_choice_count_mismatch_raises(monkeypatch):
    backend, _, _ = make_backend(monkeypatch, lambda p: [f"C{MOL_END}"])
    with pytest.raises(RuntimeError, match="choices"):
        backend.generate([MOL_START, MOL_START], {})


def test_request_batching_preserves_order(monkeypatch):
    """Chunked requests must still land in prompt order."""
    prompts = [f"{i}{MOL_START}" for i in range(5)]
    counter = {"n": 0}

    def completions(payload):
        out = []
        for _ in payload["prompt"]:
            out.append(f"M{counter['n']}{MOL_END}")
            counter["n"] += 1
        return out

    backend, _, posted = make_backend(monkeypatch, completions, request_batch_size=2)
    out = backend.generate(prompts, {})
    assert [len(p["prompt"]) for p in posted] == [2, 2, 1]
    assert out == [f"{p}M{i}{MOL_END}" for i, p in enumerate(prompts)]


def test_n_greater_than_one_keeps_first_choice_per_prompt(monkeypatch):
    """vLLM emits n consecutive choices per prompt; one per prompt comes back."""
    prompts = [f"A{MOL_START}", f"B{MOL_START}"]

    def completions(payload):
        # 2 prompts x n=2 -> indices 0,1 belong to prompt 0; 2,3 to prompt 1.
        return [f"keep0{MOL_END}", f"drop0{MOL_END}", f"keep1{MOL_END}", f"drop1{MOL_END}"]

    backend, _, _ = make_backend(monkeypatch, completions)
    out = backend.generate(prompts, {"num_return_sequences": 2})
    assert out == [f"A{MOL_START}keep0{MOL_END}", f"B{MOL_START}keep1{MOL_END}"]


def test_empty_prompt_list_is_a_no_op(monkeypatch):
    backend, _, posted = make_backend(monkeypatch, lambda p: [])
    assert backend.generate([], {}) == []
    assert posted == []


# ---------------------------------------------------------------------------
# Self-check: must tolerate sampling, must not tolerate misconfiguration
# ---------------------------------------------------------------------------


def test_self_check_passes_when_one_late_probe_closes_a_molecule(monkeypatch):
    """Regression: one unlucky sampled probe killed healthy seeds at startup."""
    draws = iter(["CC", "CCC", f"CCO{MOL_END}", f"C{MOL_END}"])

    def completions(payload):
        return [next(draws)]

    backend, _, posted = make_backend(monkeypatch, completions)
    backend.self_check()  # must not raise
    assert len(posted) == 3  # stopped as soon as a probe closed


def test_self_check_fails_when_every_probe_is_empty(monkeypatch):
    """All-empty is the missing-BOS signature and is never sampling noise.

    Note which guard actually fires: ``generate()`` rejects an all-empty batch
    before ``self_check`` ever gets to compare its probes, so the message the
    operator sees is ``generate()``'s. That one already names BOS, so the
    diagnosis is not lost -- but it does mean ``self_check``'s own
    "came back empty" branch is unreachable in practice. Asserted here so the
    arrangement is deliberate rather than discovered during an outage.
    """
    backend, _, _ = make_backend(monkeypatch, lambda p: [""])
    with pytest.raises(RuntimeError, match="empty completions"):
        backend.self_check()


def test_self_check_fails_when_no_probe_carries_the_end_tag(monkeypatch):
    """Non-empty but tagless is the skip_special_tokens signature."""
    backend, _, _ = make_backend(monkeypatch, lambda p: ["SMILESCCOSMILES"])
    with pytest.raises(RuntimeError, match="none of 4 probes"):
        backend.self_check()


def test_self_check_probe_matches_this_repos_prompt_shape(monkeypatch):
    """genetic_chemllama prompts end with mol_tags[0] and carry no leading EOS."""
    backend, tokenizer, _ = make_backend(monkeypatch, lambda p: [f"CC{MOL_END}"])
    backend.self_check()
    assert tokenizer.calls[0]["prompts"] == [MOL_START]


# ---------------------------------------------------------------------------
# Factory and helpers
# ---------------------------------------------------------------------------


def test_hf_is_the_default_and_needs_a_model():
    with pytest.raises(ValueError, match="requires model and tokenizer"):
        build_backend("hf")


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="Unknown generation backend"):
        build_backend("tensorrt")


def test_vllm_without_a_url_is_rejected(monkeypatch):
    monkeypatch.delenv("VLLM_SERVICE_URL", raising=False)
    with pytest.raises(ValueError, match="generation_backend_url"):
        build_backend("vllm", tokenizer=FakeTokenizer())


@pytest.mark.parametrize(
    "given,expected",
    [
        ("http://h:8000", "http://h:8000"),
        ("http://h:8000/", "http://h:8000"),
        ("http://h:8000/v1", "http://h:8000"),
        ("h:8000", "http://h:8000"),
        ("h:8000/v1/", "http://h:8000"),
    ],
)
def test_base_url_normalization(given, expected):
    assert VLLMServerBackend._normalize_base_url(given) == expected


class _Args:
    def __init__(self, gen_batch_size):
        self.gen_batch_size = gen_batch_size


@pytest.mark.parametrize(
    "configured,n_prompts,expected",
    [
        (32, 100, 32),     # honoured when it fits
        (None, 100, 100),  # None means one batch
        (500, 100, 100),   # clamped to the prompt count
        (0, 100, 1),       # never zero: that would loop forever
    ],
)
def test_gen_batch_size_rule_is_unchanged(configured, n_prompts, expected):
    """The HF path must batch exactly as the old inline loop did."""
    assert _resolve_gen_batch_size(_Args(configured), n_prompts) == expected
