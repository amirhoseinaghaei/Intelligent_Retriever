"""GRPO advantages, completion masking, probabilities, and loss."""
import torch


def group_advantages(rewards):
    values = torch.as_tensor(rewards, dtype=torch.float32)
    if values.ndim != 1 or len(values) < 2 or not torch.isfinite(values).all():
        raise ValueError("A group requires at least two finite rewards")
    return (values - values.mean()) / (values.std(unbiased=False) + 1e-4)


def completion_mask(ids, eos_token_id):
    """Include first EOS, exclude padding after it, even when pad == EOS."""
    if eos_token_id is None:
        return torch.ones_like(ids, dtype=torch.bool)
    eos = ids.eq(eos_token_id)
    return (eos.long().cumsum(-1) - eos.long()).eq(0)


def token_log_probs(model, ids, prompt_length):
    if not 0 < prompt_length < ids.shape[1]:
        raise ValueError("Need a prompt and at least one completion token")
    # Position p-1 predicts the first completion token at position p.
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                   use_cache=False).logits[:, prompt_length - 1:-1, :].float()
    return logits.log_softmax(-1).gather(-1, ids[:, prompt_length:].unsqueeze(-1)).squeeze(-1)


def grpo_loss(current, old, reference, mask, advantage, beta=0.04, clip=0.2):
    """Original per-sequence GRPO: clipped policy ratio and sampled reverse KL."""
    if not current.shape == old.shape == reference.shape == mask.shape:
        raise ValueError("Log probabilities and mask must have matching shapes")
    if not mask.any():
        raise ValueError("Need at least one valid completion token")
    ratio = (current - old.detach()).exp()
    surrogate = torch.minimum(ratio * advantage,
                              ratio.clamp(1 - clip, 1 + clip) * advantage)
    log_ratio = reference.detach() - current
    kl = log_ratio.exp() - log_ratio - 1
    count = mask.sum().clamp_min(1)
    return ((-surrogate + beta * kl) * mask).sum() / count, (kl.detach() * mask).sum() / count


