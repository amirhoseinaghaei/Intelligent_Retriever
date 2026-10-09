"""Generate query groups, score retrieval, and train LoRA with TRL GRPO."""
import argparse
import json
from pathlib import Path

from llm_task import completion_text, prompt_for, score_query
from retrieval import TEST, TRAIN


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--samples", type=int, default=2,
                        help="Queries sampled per question; at least two")
    parser.add_argument("--max-query-tokens", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("outputs/llm-grpo"))
    parser.add_argument("--cpu", action="store_true", help="Allow slow CPU training")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.samples < 2 or args.steps < 1 or args.max_query_tokens < 1:
        parser.error("samples must be >= 2; steps and max-query-tokens must be positive")
    return args


def evaluate(model, tokenizer, torch, max_tokens):
    records = []
    model.eval()
    with torch.no_grad():
        for question, gold in TEST:
            text = tokenizer.apply_chat_template(prompt_for(question), tokenize=False,
                                                 add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            generated = model.generate(**inputs, max_new_tokens=max_tokens,
                                       do_sample=False, use_cache=True,
                                       temperature=1.0, top_p=1.0, top_k=50,
                                       pad_token_id=tokenizer.pad_token_id)
            query = tokenizer.decode(generated[0, inputs.input_ids.shape[1]:],
                                     skip_special_tokens=True)
            records.append({"question": question, "gold": gold, **score_query(query, gold)})
    return {"recall_at_2": sum(r["recall_at_2"] for r in records) / len(records),
            "mrr_at_2": sum(r["mrr_at_2"] for r in records) / len(records),
            "records": records}


def main():
    args = parse_args()
    # Imports are delayed so --help works before installing LLM dependencies.
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise SystemExit("Install LLM dependencies: python -m pip install -r requirements-llm.txt") from exc
    cuda = torch.cuda.is_available() and not args.cpu
    if not cuda and not args.cpu:
        raise SystemExit("CUDA is unavailable. Check nvidia-smi and your PyTorch installation, "
                         "or use --cpu for a slow smoke run.")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("Output directory is nonempty; use a different --output to preserve artifacts.")
    args.output.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    bf16 = cuda and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="sdpa")
    model.to("cuda" if cuda else "cpu")
    before = evaluate(model, tokenizer, torch, args.max_query_tokens)
    (args.output / "before.json").write_text(json.dumps(before, indent=2) + "\n")
    print("Before:", {k: v for k, v in before.items() if k != "records"})
    model.config.use_cache = False
    dataset = Dataset.from_list([
        {"prompt": prompt_for(question), "question": question, "gold": gold}
        for question, gold in TRAIN
    ])
    rollout_path = args.output / "rollouts.jsonl"

    def retrieval_reward(completions, gold, question, **kwargs):
        if not len(completions) == len(gold) == len(question):
            raise ValueError("Reward inputs must have matching lengths")
        rewards = []
        state = kwargs.get("trainer_state")
        with rollout_path.open("a") as log:
            for completion, label, q in zip(completions, gold, question):
                result = score_query(completion_text(completion), label)
                rewards.append(result["reward"])
                log.write(json.dumps({"step": getattr(state, "global_step", None),
                                      "question": q, "gold": label, **result}) + "\n")
        return rewards

    config = GRPOConfig(
        output_dir=str(args.output), max_steps=args.steps,
        per_device_train_batch_size=1, gradient_accumulation_steps=args.samples,
        num_generations=args.samples, generation_batch_size=args.samples,
        max_prompt_length=256, max_completion_length=args.max_query_tokens,
        temperature=0.9, top_p=0.95, learning_rate=5e-5,
        beta=0.04, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=bf16, fp16=False, use_cpu=args.cpu,
        use_vllm=False, optim="adamw_torch", report_to="none",
        logging_steps=1, save_strategy="no", seed=args.seed,
    )
    trainer = GRPOTrainer(
        model=model, args=config, train_dataset=dataset,
        processing_class=tokenizer, reward_funcs=retrieval_reward,
        peft_config=LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                               target_modules=["q_proj", "v_proj"],
                               task_type="CAUSAL_LM", bias="none"),
    )
    trainable = [(name, p) for name, p in trainer.model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise RuntimeError("Expected only LoRA parameters to be trainable")
    # Keep the small adapter tensors to verify that optimization changed weights.
    initial = {name: p.detach().cpu().clone() for name, p in trainable}
    trainer.model.print_trainable_parameters()
    trainer.train()
    changed = any(not torch.equal(initial[name], p.detach().cpu()) for name, p in trainable)
    adapter = args.output / "adapter"
    trainer.save_model(str(adapter))
    tokenizer.save_pretrained(str(adapter))
    after = evaluate(trainer.model, tokenizer, torch, args.max_query_tokens)
    (args.output / "after.json").write_text(json.dumps(after, indent=2) + "\n")
    summary = {"model": args.model, "steps": args.steps, "samples": args.samples,
               "seed": args.seed, "adapter_weights_changed": changed,
               "before": {k: v for k, v in before.items() if k != "records"},
               "after": {k: v for k, v in after.items() if k != "records"}}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("Saved LoRA adapter:", adapter)
    if not changed:
        print("No adapter weights changed. Check rollouts: equal rewards within a group "
              "produce no retrieval-learning signal; try more samples or more varied questions.")


if __name__ == "__main__":
    main()
