"""Generate a query with a saved LoRA adapter and retrieve matching documents."""
import argparse

from llm_task import prompt_for
from retrieval import DOCUMENTS, retrieve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--adapter", default="outputs/llm-grpo/adapter")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer
    cuda = torch.cuda.is_available() and not args.cpu
    dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    model = AutoPeftModelForCausalLM.from_pretrained(args.adapter, torch_dtype=dtype)
    model.to("cuda" if cuda else "cpu").eval()
    text = tokenizer.apply_chat_template(prompt_for(args.question), tokenize=False,
                                         add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=32, do_sample=False,
                                   temperature=1.0, top_p=1.0, top_k=50,
                                   pad_token_id=tokenizer.pad_token_id)
    query = tokenizer.decode(generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print("Query:", query)
    ranked, scores = retrieve(query, action=0)
    for doc in ranked:
        if scores[doc] > 0:
            print(f"{doc} (score={scores[doc]:.3f}): {DOCUMENTS[doc]}")


if __name__ == "__main__":
    main()
