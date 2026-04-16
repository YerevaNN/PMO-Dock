from typing import List
import gc
import torch

from logging_ import logger


def generate(
    model,
    tokenizer,
    prompts: List[str],
    gen_batch_size,
    mol_end_token_id,
    pad_token_id,
    gen_config,
    device
):
    if len(prompts) == 0:
        return {}

    generation_dict = {submol: [] for submol in prompts}
    num_return_sequences = gen_config['num_return_sequences']
    data = None
    outputs = None
    try:
        submols_length = len(prompts)
        for start_idx in range(0, submols_length, gen_batch_size):
            end_idx = min(start_idx + gen_batch_size, submols_length)
            current_submols = prompts[start_idx:end_idx]
            data = tokenizer(current_submols, return_tensors="pt", padding=True, add_special_tokens=True).to(device)
            outputs = model.generate(
                **data,
                **gen_config,
                eos_token_id=mol_end_token_id,
                pad_token_id=pad_token_id,
                cache_implementation="static"
            )
            generated_mols = tokenizer.batch_decode(outputs, add_special_tokens=False)
            for i, submol in enumerate(current_submols):
                generation_dict[submol].extend(generated_mols[i*num_return_sequences:(i+1)*num_return_sequences])
    except Exception as e:
        logger.info(e)
        raise e
    finally:
        # clean up
        del data
        del outputs
        gc.collect()
        torch.cuda.empty_cache()

    return generation_dict


def perplexity(
    logits: torch.tensor,
    labels: torch.tensor,
    bos_token_id: int,
    pad_token_id: int
):
    mask = (labels == pad_token_id) | (labels == bos_token_id) # create a mask for not exluding eos and pad_tokens
    log_probs = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), labels.flatten(0, 1), reduction="none"
    ) / torch.log(torch.tensor([2], device=logits.device)) # calculate log (2 based) probs
    log_probs = log_probs.view(labels.shape)
    log_probs[mask] = 0 # exclude pad and bos token probs
    log_probs = log_probs.sum(1) / (log_probs != 0).sum(-1)
    return 2 ** log_probs


def model_perplexity(
    model,
    tokenizer,
    prompts: List[str],
    batch_size,
    bos_token_id,
    pad_token_id,
    device
):
    if len(prompts) == 0:
        return {}

    perplexity_dict = {}
    data = None
    outputs = None
    try:
        submols_length = len(prompts)
        for start_idx in range(0, submols_length, batch_size):
            end_idx = min(start_idx + batch_size, submols_length)
            current_prompts = prompts[start_idx:end_idx]
            data = tokenizer(current_prompts, return_tensors="pt", padding=True, add_special_tokens=True).to(device)
            outputs = model(
                input_ids=data.input_ids[:, :-1],
                attention_mask=data.attention_mask[:, 1:],
            )
            logits = outputs.logits.detach().clone()
            labels = data.input_ids[:, 1:]
            perplexities = perplexity(
                logits, labels,
                bos_token_id=bos_token_id,
                pad_token_id=pad_token_id
            )
            for i, prompt in enumerate(current_prompts):
                perplexity_dict[prompt] = perplexities[i].item()
    except Exception as e:
        logger.info(e)
        raise e
    finally:
        # clean up
        del data
        del outputs
        gc.collect()
        torch.cuda.empty_cache()

    return perplexity_dict
