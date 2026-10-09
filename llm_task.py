"""Prompt and retrieval reward shared by training, evaluation, and tests."""
from retrieval import DOCUMENTS, retrieve, tokens

SYSTEM = (
    "Rewrite the user's question as a search query for an operations knowledge base "
    "covering credentials, backups, billing, API limits, deployments, and logs. "
    "Return only one line containing 1 to 12 search words. "
    "Do not answer the question or include explanations."
)


def prompt_for(question):
    # No gold document ID or gold text is exposed to the model.
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": question}]


def completion_text(completion):
    if isinstance(completion, str):
        return completion
    return "".join(message.get("content", "") for message in completion)


def score_query(query, gold):
    if gold not in DOCUMENTS:
        raise ValueError(f"Unknown gold document: {gold}")
    query = query.strip()
    words = tokens(query)
    valid = 1 <= len(words) <= 12 and "\n" not in query and "\r" not in query
    ranked, scores = retrieve(query, action=0)
    # Ignore zero-overlap results, including alphabetically selected ties.
    hits = [doc for doc in ranked if scores[doc] > 0]
    rr = 1 / (hits.index(gold) + 1) if valid and gold in hits else 0.0
    return {"query": query, "ranked": hits, "scores": scores,
            "valid": valid, "recall_at_2": float(valid and gold in hits),
            "mrr_at_2": rr, "reward": rr if valid else -1.0}
