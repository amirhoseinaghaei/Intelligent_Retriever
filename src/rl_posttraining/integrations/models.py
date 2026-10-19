"""Construct the tokenizer, frozen base model, and trainable LoRA learner."""


def load_learner(config):
    # Preparation and CLI help must work without importing training dependencies.
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    from rl_posttraining.core.grpo.learner import QueryLearner
    from rl_posttraining.core.investigate import prompt_for

    set_seed(config["seed"])
    torch.set_num_threads(1)
    cuda = not config["cpu"]
    if cuda and not torch.cuda.is_available():
        raise RuntimeError("Learner cannot access CUDA; check PyTorch/driver or use --cpu")
    dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(config["model"], padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        config["model"], torch_dtype=dtype, attn_implementation="sdpa")
    model = get_peft_model(base, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM", bias="none"))
    model.to("cuda" if cuda else "cpu")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.print_trainable_parameters()
    return QueryLearner(model, tokenizer, config["learning_rate"], config["beta"],
                        config["clip"], config["update_epochs"], prompt_builder=prompt_for)
