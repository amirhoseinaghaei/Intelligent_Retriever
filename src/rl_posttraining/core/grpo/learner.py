"""LoRA query generation and GRPO optimization."""
import math

import torch

from .loss import completion_mask, group_advantages, grpo_loss, token_log_probs


class QueryLearner:
    """One PEFT model, with serial generation and microbatched optimization."""

    def __init__(self, model, tokenizer, learning_rate=1e-5, beta=0.04,
                 clip=0.2, update_epochs=1, *, prompt_builder):
        self.model, self.tokenizer = model, tokenizer
        self.beta, self.clip, self.update_epochs = beta, clip, update_epochs
        # Old/current/reference forwards must have identical dropout behavior.
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        self.trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if not self.trainable or any("lora_" not in n for n, _ in self.trainable):
            raise ValueError("Only LoRA parameters should be trainable")
        self.initial = {n: p.detach().cpu().clone() for n, p in self.trainable}
        self.optimizer = torch.optim.AdamW([p for _, p in self.trainable],
                                           lr=learning_rate, weight_decay=0.0)
        self.pending = None
        self.prompt_builder = prompt_builder

    def encode(self, question):
        text = self.tokenizer.apply_chat_template(self.prompt_builder(question), tokenize=False,
                                                  add_generation_prompt=True)
        return self.tokenizer(text, return_tensors="pt", return_token_type_ids=False).to(self.model.device)

    def sample(self, question, samples, max_tokens):
        if self.pending is not None:
            raise RuntimeError("Update the pending group before sampling again")
        inputs = self.encode(question)
        self.model.eval()
        with torch.no_grad():
            # Unfiltered temperature-1 sampling matches probabilities in the loss.
            generated = self.model.generate(
                **inputs, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                num_return_sequences=samples, max_new_tokens=max_tokens,
                use_cache=True, pad_token_id=self.tokenizer.pad_token_id)
        return self.prepare_group(generated, inputs.input_ids.shape[1])

    def prepare_group(self, generated, prompt_length):
        """Cache behavior/reference probabilities before any optimizer update."""
        if self.pending is not None:
            raise RuntimeError("A rollout group is already pending")
        masks = completion_mask(generated[:, prompt_length:], self.tokenizer.eos_token_id)
        self.model.train()
        records, queries = [], []
        with torch.no_grad():
            for sequence, mask in zip(generated, masks):
                ids = sequence[:prompt_length + int(mask.sum())].unsqueeze(0)
                old = token_log_probs(self.model, ids, prompt_length).detach()
                # The frozen base is the reference: no second model allocation.
                with self.model.disable_adapter():
                    ref = token_log_probs(self.model, ids, prompt_length).detach()
                records.append({"ids": ids, "old": old, "ref": ref,
                                "mask": torch.ones_like(old, dtype=torch.bool)})
                queries.append(self.tokenizer.decode(ids[0, prompt_length:], skip_special_tokens=True))
        self.pending = (prompt_length, records)
        return queries

    def update(self, rewards):
        if self.pending is None:
            raise RuntimeError("Sample a group before updating")
        prompt_length, records = self.pending
        if len(rewards) != len(records):
            raise ValueError("One reward required per generated query")
        advantages = group_advantages(rewards).to(self.model.device)
        total_loss = total_kl = 0.0
        self.model.train()
        for _ in range(self.update_epochs):
            self.optimizer.zero_grad(set_to_none=True)
            for record, advantage in zip(records, advantages):
                current = token_log_probs(self.model, record["ids"], prompt_length)
                loss, kl = grpo_loss(current, record["old"], record["ref"],
                                     record["mask"], advantage, self.beta, self.clip)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite GRPO loss")
                # One sequence per microbatch; each query has equal group weight.
                (loss / len(records)).backward()
                total_loss += loss.detach().item() / len(records)
                total_kl += kl.item() / len(records)
            grad_norm = torch.nn.utils.clip_grad_norm_([p for _, p in self.trainable], 1.0)
            if not math.isfinite(float(grad_norm)):
                raise RuntimeError("Nonfinite adapter gradient norm")
            self.optimizer.step()
        self.pending = None
        return {"loss": total_loss / self.update_epochs,
                "kl": total_kl / self.update_epochs,
                "mean_reward": sum(rewards) / len(rewards),
                "reward_std": float(torch.tensor(rewards).float().std(unbiased=False)),
                "advantages": advantages.cpu().tolist(), "grad_norm": float(grad_norm)}

    def save(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        return any(not torch.equal(self.initial[n], p.detach().cpu()) for n, p in self.trainable)


