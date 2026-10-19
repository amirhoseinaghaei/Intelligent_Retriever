"""Offline GRPO checks, including a real tiny Qwen + PEFT backward pass."""
import copy
import tempfile
import unittest

import torch

from rl_posttraining.core.grpo.learner import QueryLearner
from rl_posttraining.core.grpo.loss import completion_mask, group_advantages, grpo_loss, token_log_probs
from rl_posttraining.core.investigate import prompt_for


class GRPOMathTests(unittest.TestCase):
    def test_group_advantages_and_equal_rewards(self):
        self.assertTrue(torch.allclose(group_advantages([0, 1]), torch.tensor([-1., 1.]), atol=0.001))
        self.assertTrue(torch.equal(group_advantages([1, 1]), torch.zeros(2)))
        with self.assertRaises(ValueError):
            group_advantages([float("nan"), 1])

    def test_eos_mask_includes_first_eos_only(self):
        ids = torch.tensor([[4, 2, 2, 2], [4, 5, 6, 7], [2, 2, 2, 2]])
        expected = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]]).bool()
        self.assertTrue(torch.equal(completion_mask(ids, 2), expected))

    def test_reward_sign_and_masked_token_gradients(self):
        for advantage, expected_sign in [(1., -1.), (-1., 1.)]:
            current = torch.tensor([[-2., -2.]], requires_grad=True)
            old = current.detach().clone()
            loss, _ = grpo_loss(current, old, old, torch.tensor([[True, False]]), advantage, beta=0)
            loss.backward()
            self.assertEqual(float(current.grad[0, 0]), expected_sign)
            self.assertEqual(float(current.grad[0, 1]), 0)

    def test_clipped_ratio_and_reference_kl(self):
        current = torch.tensor([[0.4]], requires_grad=True)
        old = torch.zeros_like(current)
        loss, _ = grpo_loss(current, old, old, torch.ones_like(current).bool(), 1., beta=0)
        self.assertAlmostEqual(loss.item(), -1.2, places=5)
        loss.backward()
        self.assertEqual(current.grad.item(), 0)
        current = torch.tensor([[-2.]], requires_grad=True)
        ref = torch.tensor([[-1.]])
        loss, kl = grpo_loss(current, current.detach(), ref,
                             torch.ones_like(current).bool(), 0., beta=0.04)
        self.assertGreater(kl.item(), 0)
        loss.backward()
        self.assertLess(current.grad.item(), 0)  # Descent increases log probability toward ref.


class TinyQwenIntegrationTests(unittest.TestCase):
    def make_learner(self):
        from peft import LoraConfig, get_peft_model
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
        torch.manual_seed(7)
        torch.set_num_threads(1)
        vocab = {"[UNK]": 0, "[BOS]": 1, "[EOS]": 2, "question": 3,
                 "backup": 4, "snapshots": 5, "billing": 6, "invoices": 7}
        raw = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        raw.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]",
                                            bos_token="[BOS]", eos_token="[EOS]", pad_token="[EOS]")
        tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}"
        base = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=8, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=256,
            eos_token_id=2, pad_token_id=2, bos_token_id=1))
        model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4,
                                               target_modules=["q_proj", "v_proj"],
                                               task_type="CAUSAL_LM"))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return QueryLearner(model, tokenizer, learning_rate=0.01, beta=0.04, update_epochs=2, prompt_builder=prompt_for)

    def test_nonzero_rewards_update_only_lora_and_reload_matches(self):
        from peft import PeftModel
        from transformers import Qwen2ForCausalLM
        learner = self.make_learner()
        model = learner.model
        base_before = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
        # Fixed rewards isolate optimizer behavior from external retrieval services.
        generated = torch.tensor([[1, 3, 4, 5, 2, 2], [1, 3, 6, 7, 2, 2]])
        queries = learner.prepare_group(generated, prompt_length=2)
        self.assertEqual(len(queries), 2)
        rewards = [1.0, 0.0]
        records = learner.pending[1]
        self.assertEqual(records[0]["old"].shape, (1, 3))
        self.assertFalse(records[0]["old"].requires_grad)
        metrics = learner.update(rewards)
        self.assertGreater(metrics["grad_norm"], 0)
        self.assertLess(metrics["advantages"][1], 0)
        for n, p in model.named_parameters():
            if n in base_before:
                self.assertTrue(torch.equal(base_before[n], p), n)
        with tempfile.TemporaryDirectory() as tmp:
            # Save base and adapter separately to test actual PEFT reload.
            base_path, adapter_path = tmp + "/base", tmp + "/adapter"
            with model.disable_adapter():
                reference = token_log_probs(model, generated[:1, :5], 2).detach()
            copy.deepcopy(model).unload().save_pretrained(base_path)
            self.assertTrue(learner.save(adapter_path))
            reloaded = PeftModel.from_pretrained(Qwen2ForCausalLM.from_pretrained(base_path), adapter_path)
            with torch.no_grad():
                actual = token_log_probs(reloaded, generated[:1, :5], 2)
                expected = token_log_probs(model, generated[:1, :5], 2)
            self.assertTrue(torch.allclose(actual, expected, atol=1e-5))
            self.assertFalse(torch.allclose(actual, reference, atol=1e-5))

    def test_actual_generation_and_equal_reward_group(self):
        learner = self.make_learner()
        queries = learner.sample("question", samples=2, max_tokens=3)
        self.assertEqual(len(queries), 2)
        metrics = learner.update([0., 0.])
        self.assertEqual(metrics["reward_std"], 0)
        self.assertEqual(metrics["advantages"], [0., 0.])
        for name, param in learner.trainable:
            self.assertTrue(torch.equal(learner.initial[name], param.detach()), name)


if __name__ == "__main__":
    unittest.main()
